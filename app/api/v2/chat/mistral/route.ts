import { getAIProfile } from "@/lib/server/server-chat-helpers"
import { ServerRuntime } from "next"

import { updateOrAddSystemMessage } from "@/lib/ai-helper"

import llmConfig from "@/lib/models/llm/llm-config"
import { checkRatelimitOnApi } from "@/lib/server/ratelimiter"
import {
  buildFinalMessages,
  filterEmptyAssistantMessages,
  handleAssistantMessages
} from "@/lib/build-prompt"
import {
  handleErrorResponse,
  handleOpenRouterApiError
} from "@/lib/models/llm/api-error"
import { generateStandaloneQuestion } from "@/lib/models/question-generator"
import { getSelectedModel } from "@/lib/models/notdiamond"

interface RequestBody {
  model: string | undefined
  route: string
  messages: { role: any; content: any }[]
  temperature: number
  max_tokens: number
  stream: boolean
  stop?: string[]
  provider?: {
    order: string[]
  }
}

export const runtime: ServerRuntime = "edge"
export const preferredRegion = [
  "iad1",
  "arn1",
  "bom1",
  "cdg1",
  "cle1",
  "cpt1",
  "dub1",
  "fra1",
  "gru1",
  "hnd1",
  "icn1",
  "kix1",
  "lhr1",
  "pdx1",
  "sfo1",
  "sin1",
  "syd1"
]

export async function POST(request: Request) {
  const json = await request.json()
  const {
    payload,
    chatImages,
    selectedPlugin,
    detectedModerationLevel,
    isRetrieval,
    isContinuation
  } = json as {
    payload: any
    chatImages: any
    selectedPlugin: any
    detectedModerationLevel: number
    isRetrieval: boolean
    isContinuation: boolean
  }

  const isRagEnabled = json.isRagEnabled ?? true
  let ragUsed = false
  let ragId: string | null = null
  const shouldUseRAG = !isRetrieval && isRagEnabled

  try {
    const profile = await getAIProfile()
    const chatSettings = payload.chatSettings

    let {
      providerUrl,
      providerBaseUrl,
      providerHeaders,
      selectedModel,
      selectedStandaloneQuestionModel,
      rateLimitCheckResult,
      providerRouting,
      similarityTopK,
      modelTemperature,
      isPentestGPTPro
    } = await getProviderConfig(chatSettings, profile)

    if (rateLimitCheckResult !== null) {
      return rateLimitCheckResult.response
    }

    const cleanedMessages = (await buildFinalMessages(
      payload,
      profile,
      chatImages,
      selectedPlugin,
      shouldUseRAG
    )) as any[]

    updateOrAddSystemMessage(
      cleanedMessages,
      llmConfig.systemPrompts.pentestGPTChat
    )

    // On normal chat, the last user message is the target standalone message
    // On continuation, the tartget is the last generated message by the system
    const targetStandAloneMessage =
      cleanedMessages[cleanedMessages.length - 2].content
    const filterTargetMessage = isContinuation
      ? cleanedMessages[cleanedMessages.length - 3]
      : cleanedMessages[cleanedMessages.length - 2]

    if (
      shouldUseRAG &&
      llmConfig.hackerRAG.enabled &&
      llmConfig.hackerRAG.endpoint &&
      llmConfig.hackerRAG.apiKey &&
      cleanedMessages.length > 0 &&
      filterTargetMessage.role === "user" &&
      filterTargetMessage.content.length > llmConfig.hackerRAG.messageLength.min
    ) {
      const { standaloneQuestion, atomicQuestions } =
        await generateStandaloneQuestion(
          cleanedMessages,
          targetStandAloneMessage,
          providerBaseUrl,
          providerHeaders,
          selectedStandaloneQuestionModel,
          llmConfig.systemPrompts.pentestgptCurrentDateOnly,
          true,
          similarityTopK
        )

      const response = await fetch(llmConfig.hackerRAG.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${llmConfig.hackerRAG.apiKey}`
        },
        body: JSON.stringify({
          query: standaloneQuestion,
          questions: atomicQuestions,
          chunks: similarityTopK
        })
      })

      const data = await response.json()

      if (data && data.content) {
        ragUsed = true
        cleanedMessages[0].content =
          `${llmConfig.systemPrompts.RAG}\n` +
          `Context for RAG enrichment:\n` +
          `---------------------\n` +
          `${data.content}\n` +
          `---------------------\n` +
          `DON'T MENTION OR REFERENCE ANYTHING RELATED TO RAG CONTENT OR ANYTHING RELATED TO RAG. USER DOESN'T HAVE DIRECT ACCESS TO THIS CONTENT, ITS PURPOSE IS TO ENRICH YOUR OWN KNOWLEDGE. ROLE PLAY.`
      }
      ragId = data?.resultId
    }

    if (
      detectedModerationLevel === 1 ||
      (detectedModerationLevel > 0 && detectedModerationLevel <= 0.1) ||
      (detectedModerationLevel >= 0.9 && detectedModerationLevel < 1)
    ) {
      filterEmptyAssistantMessages(cleanedMessages)
    } else if (
      detectedModerationLevel === -1 ||
      (detectedModerationLevel > 0.1 && detectedModerationLevel < 0.9)
    ) {
      handleAssistantMessages(cleanedMessages)
    } else {
      filterEmptyAssistantMessages(cleanedMessages)
    }

    if (isPentestGPTPro && !ragUsed) {
      selectedModel = await getSelectedModel(cleanedMessages, "pentestgpt4")
    }

    if (selectedModel === "meta-llama/llama-3.1-70b-instruct") {
      providerRouting = {
        order: ["OctoAI"]
      }
    } else if (selectedModel === "meta-llama/llama-3.1-405b-instruct") {
      providerRouting = {
        order: ["Fireworks", "OctoAI", "Together"]
      }
    }

    const requestBody: RequestBody = {
      model: selectedModel,
      route: "fallback",
      messages: cleanedMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      temperature: modelTemperature,
      max_tokens: 1024,
      stream: true
    }

    if (providerRouting) {
      requestBody.provider = providerRouting
    }

    try {
      const res = await fetch(providerUrl, {
        method: "POST",
        headers: providerHeaders,
        body: JSON.stringify(requestBody)
      })

      if (!res.ok) {
        await handleOpenRouterApiError(res)
      }

      if (!res.body) {
        throw new Error("Response body is null")
      }

      // Create a new ReadableStream to combine additional data with the original stream
      const combinedStream = new ReadableStream({
        async start(controller) {
          // Add additional information at the beginning
          const additionalInfo = JSON.stringify({ ragUsed, ragId })
          controller.enqueue(
            new TextEncoder().encode(`RAG: ${additionalInfo}\n\n`)
          )

          // Pipe the original response stream
          const reader = res.body?.getReader()
          while (reader) {
            const { done, value } = await reader.read()
            if (done) break
            controller.enqueue(value)
          }

          controller.close()
        }
      })

      return new Response(combinedStream, {
        headers: res.headers
      })
    } catch (error) {
      return handleErrorResponse(error)
    }
  } catch (error: any) {
    const errorMessage = error.message || "An unexpected error occurred"
    const errorCode = error.status || 500

    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}

async function getProviderConfig(chatSettings: any, profile: any) {
  const isPentestGPTPro = chatSettings.model === "mistral-large"

  const defaultModel = llmConfig.models.pentestgpt_default_openrouter
  const proModel = llmConfig.models.pentestgpt_pro_openrouter
  const selectedStandaloneQuestionModel =
    llmConfig.models.pentestgpt_standalone_question_openrouter

  const providerUrl = llmConfig.openrouter.url
  const providerBaseUrl = llmConfig.openrouter.baseUrl

  const providerHeaders = {
    Authorization: `Bearer ${llmConfig.openrouter.apiKey}`,
    "Content-Type": "application/json",
    "HTTP-Referer": `https://pentestgpt.com/${chatSettings.model}`,
    "X-Title": chatSettings.model
  }

  let modelTemperature = 0.4
  let similarityTopK = 3
  let selectedModel = isPentestGPTPro ? proModel : defaultModel
  let rateLimitCheckResult = await checkRatelimitOnApi(
    profile.user_id,
    isPentestGPTPro ? "pentestgpt-pro" : "pentestgpt"
  )

  let providerRouting
  if (process.env.OPENROUTER_FIRST_PROVIDER) {
    providerRouting = llmConfig.openrouter.providerRouting
  }

  if (isPentestGPTPro) {
    providerRouting = {
      order: ["Together", "OctoAI"]
    }
  }

  if (selectedModel === "mistralai/mistral-nemo") {
    providerRouting = {
      order: ["Lepton", "Mistral"]
    }
    modelTemperature = 0.3
  }

  return {
    providerUrl,
    providerBaseUrl,
    providerHeaders,
    selectedModel,
    selectedStandaloneQuestionModel,
    rateLimitCheckResult,
    providerRouting,
    similarityTopK,
    isPentestGPTPro,
    modelTemperature
  }
}
