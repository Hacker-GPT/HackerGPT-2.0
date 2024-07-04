import { getServerProfile } from "@/lib/server/server-chat-helpers"
import { ServerRuntime } from "next"

import { updateOrAddSystemMessage } from "@/lib/ai-helper"
import RetrieverReranker from "@/lib/models/query-pinecone-2v"

import llmConfig from "@/lib/models/llm/llm-config"
import { checkRatelimitOnApi } from "@/lib/server/ratelimiter"
import {
  buildFinalMessages,
  ensureAssistantMessagesNotEmpty,
  filterEmptyAssistantMessages
} from "@/lib/build-prompt"
import {
  handleErrorResponse,
  handleOpenRouterApiError
} from "@/lib/models/llm/api-error"
import { generateStandaloneQuestion } from "@/lib/models/question-generator"
import { isPremiumUser } from "@/lib/server/subscription-utils"
import { PluginID } from "@/types/plugins"

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

export async function POST(request: Request) {
  const json = await request.json()
  const { payload, chatImages, selectedPlugin, isRetrieval, isContinuation } =
    json as {
      payload: any
      chatImages: any
      selectedPlugin: any
      isRetrieval: boolean
      isContinuation: boolean
    }

  const isRagEnabled = json.isRagEnabled ?? true
  const shouldUseRAG =
    !isRetrieval && isRagEnabled && selectedPlugin !== PluginID.CODE_LLM

  try {
    const profile = await getServerProfile()
    const chatSettings = payload.chatSettings

    const isPremium = await isPremiumUser(profile.user_id)

    let {
      useOpenRouter,
      providerUrl,
      providerHeaders,
      selectedModel,
      selectedStandaloneQuestionModel,
      stopSequence,
      rateLimitCheckResult,
      providerRouting,
      similarityTopK,
      isMistralLarge,
      modelTemperature
    } = await getProviderConfig(
      chatSettings,
      profile,
      isPremium,
      selectedPlugin
    )

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

    const systemMessageContent = `${llmConfig.systemPrompts.hackerGPT}`
    updateOrAddSystemMessage(cleanedMessages, systemMessageContent)

    // On normal chat, the last user message is the target standalone message
    // On continuation, the tartget is the last generated message by the system
    const targetStandAloneMessage =
      cleanedMessages[cleanedMessages.length - 2].content
    const filterTargetMessage = isContinuation
      ? cleanedMessages[cleanedMessages.length - 3]
      : cleanedMessages[cleanedMessages.length - 2]

    if (shouldUseRAG) {
      if (
        !llmConfig.hackerRAG.enabled &&
        llmConfig.usePinecone &&
        cleanedMessages.length > 0 &&
        filterTargetMessage.role === "user" &&
        filterTargetMessage.content.length >
          llmConfig.pinecone.messageLength.min &&
        filterTargetMessage.content.length <
          llmConfig.pinecone.messageLength.max
      ) {
        const { standaloneQuestion } = await generateStandaloneQuestion(
          cleanedMessages,
          targetStandAloneMessage,
          providerUrl,
          providerHeaders,
          selectedStandaloneQuestionModel,
          systemMessageContent
        )

        const pineconeRetriever = new RetrieverReranker(
          llmConfig.openai.apiKey,
          llmConfig.pinecone,
          llmConfig.cohere,
          similarityTopK
        )

        const pineconeResults =
          await pineconeRetriever.retrieve(standaloneQuestion)

        if (pineconeResults !== "None") {
          modelTemperature = llmConfig.pinecone.temperature

          selectedModel = useOpenRouter
            ? llmConfig.models.hackerGPT_RAG_openrouter
            : llmConfig.models.hackerGPT_RAG_together

          cleanedMessages[0].content =
            `${llmConfig.systemPrompts.pinecone}\n` +
            `Context for RAG enrichment:\n` +
            `---------------------\n` +
            `${pineconeResults}\n` +
            `---------------------\n` +
            `DON'T MENTION OR REFERENCE ANYTHING RELATED TO RAG CONTENT OR ANYTHING RELATED TO RAG. USER DOESN'T HAVE DIRECT ACCESS TO THIS CONTENT, ITS PURPOSE IS TO ENRICH YOUR OWN KNOWLEDGE. ROLE PLAY.`
        }
      } else if (
        llmConfig.hackerRAG.enabled &&
        llmConfig.hackerRAG.endpoint &&
        llmConfig.hackerRAG.apiKey &&
        cleanedMessages.length > 0 &&
        filterTargetMessage.role === "user" &&
        filterTargetMessage.content.length >
          llmConfig.pinecone.messageLength.min &&
        filterTargetMessage.content.length <
          llmConfig.pinecone.messageLength.max
      ) {
        // Hacker RAG Implementation
        const { standaloneQuestion, atomicQuestions } =
          await generateStandaloneQuestion(
            cleanedMessages,
            targetStandAloneMessage,
            providerUrl,
            providerHeaders,
            selectedStandaloneQuestionModel,
            systemMessageContent,
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
          cleanedMessages[0].content =
            `${llmConfig.systemPrompts.pinecone}\n` +
            `Context for RAG enrichment:\n` +
            `---------------------\n` +
            `${data.content}\n` +
            `---------------------\n` +
            `DON'T MENTION OR REFERENCE ANYTHING RELATED TO RAG CONTENT OR ANYTHING RELATED TO RAG. USER DOESN'T HAVE DIRECT ACCESS TO THIS CONTENT, ITS PURPOSE IS TO ENRICH YOUR OWN KNOWLEDGE. ROLE PLAY.`
        }
      }
    }

    // If the user is using the mistral-large model, we must switch to the pro model.
    if (isMistralLarge && !isRagEnabled) {
      selectedModel = useOpenRouter
        ? llmConfig.models.hackerGPT_pro_openrouter
        : llmConfig.models.hackerGPT_pro_together
    }

    if (
      (isPremium && !isMistralLarge) ||
      selectedPlugin === PluginID.CODE_LLM
    ) {
      ensureAssistantMessagesNotEmpty(cleanedMessages)
    } else {
      filterEmptyAssistantMessages(cleanedMessages)
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

    if (stopSequence && stopSequence.length > 0) {
      requestBody.stop = stopSequence
    }

    if (providerRouting && selectedPlugin !== PluginID.CODE_LLM) {
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
      return res
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

async function getProviderConfig(
  chatSettings: any,
  profile: any,
  isPremium: boolean,
  selectedPlugin: string
) {
  const useOpenRouter = llmConfig.useOpenRouter
  const isMistralLarge = chatSettings.model === "mistral-large"

  const providerConfig = useOpenRouter
    ? llmConfig.openrouter
    : llmConfig.together
  const apiKey = useOpenRouter
    ? llmConfig.openrouter.apiKey
    : llmConfig.together.apiKey
  const defaultModel = useOpenRouter
    ? llmConfig.models.hackerGPT_default_openrouter
    : llmConfig.models.hackerGPT_default_together
  const proModel = useOpenRouter
    ? llmConfig.models.hackerGPT_pro_openrouter
    : llmConfig.models.hackerGPT_pro_together
  const selectedStandaloneQuestionModel = useOpenRouter
    ? llmConfig.models.hackerGPT_standalone_question_openrouter
    : llmConfig.models.hackerGPT_standalone_question_together

  const providerUrl = providerConfig.url
  const providerHeaders = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json"
  }

  let modelTemperature = 0.4
  let similarityTopK = 3
  let selectedModel = isMistralLarge ? proModel : defaultModel
  let stopSequence = !useOpenRouter ? ["[/INST]", "</s>"] : undefined
  let rateLimitCheckResult = await checkRatelimitOnApi(
    profile.user_id,
    isMistralLarge ? "hackergpt-pro" : "hackergpt"
  )

  let providerRouting
  if (process.env.OPENROUTER_FIRST_PROVIDER && useOpenRouter) {
    providerRouting = llmConfig.openrouter.providerRouting
  }

  if (isPremium && !isMistralLarge) {
    selectedModel = "meta-llama/llama-3-8b-instruct"
    providerRouting = {
      order: [`Together`]
    }
  }

  if (selectedPlugin === PluginID.CODE_LLM) {
    selectedModel = "deepseek/deepseek-coder"
  }

  return {
    useOpenRouter,
    providerUrl,
    providerHeaders,
    selectedModel,
    selectedStandaloneQuestionModel,
    stopSequence,
    rateLimitCheckResult,
    providerRouting,
    similarityTopK,
    isMistralLarge,
    modelTemperature
  }
}
