import { getServerProfile } from "@/lib/server/server-chat-helpers"
import { ServerRuntime } from "next"

import { updateOrAddSystemMessage } from "@/lib/ai-helper"

import llmConfig from "@/lib/models/llm/llm-config"
import { checkRatelimitOnApi } from "@/lib/server/ratelimiter"
import {
  buildFinalMessages,
  ensureAssistantMessagesNotEmpty
} from "@/lib/build-prompt"
import { APIError, handleOpenRouterApiError } from "@/lib/models/llm/api-error"
import { GPT4 } from "@/lib/models/llm/openai-llm-list"

interface RequestBody {
  model: string | undefined
  route: string
  messages: { role: any; content: any }[]
  temperature: number
  max_tokens: number
  stream: boolean
  provider?: {
    order: string[]
  }
}

export const runtime: ServerRuntime = "edge"

export async function POST(request: Request) {
  const json = await request.json()
  const { payload, chatImages, selectedPlugin } = json as {
    payload: any
    chatImages: any
    selectedPlugin: any
  }

  try {
    const profile = await getServerProfile()
    const chatSettings = payload.chatSettings

    let {
      providerUrl,
      providerHeaders,
      selectedModel,
      rateLimitCheckResult,
      providerRouting,
      modelTemperature
    } = await getProviderConfig(chatSettings, profile)

    if (rateLimitCheckResult !== null) {
      return rateLimitCheckResult.response
    }

    const cleanedMessages = (await buildFinalMessages(
      payload,
      profile,
      chatImages,
      selectedPlugin
    )) as any[]

    const systemMessageContent =
      llmConfig.systemPrompts.hackerGPTCurrentDateOnly

    updateOrAddSystemMessage(cleanedMessages, systemMessageContent)

    ensureAssistantMessagesNotEmpty(cleanedMessages, false)

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
      return res
    } catch (error) {
      if (error instanceof APIError) {
        console.error(
          `API Error - Code: ${error.code}, Message: ${error.message}`
        )
      } else if (error instanceof Error) {
        console.error(`Unexpected Error: ${error.message}`)
      } else {
        console.error(`An unknown error occurred: ${error}`)
      }
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
  const isProModel =
    chatSettings.model === "mistral-large" ||
    chatSettings.model === GPT4.modelId

  const providerConfig = llmConfig.openrouter
  const apiKey = llmConfig.openrouter.apiKey
  const defaultModel = "perplexity/llama-3-sonar-small-32k-online"
  const proModel = "perplexity/llama-3-sonar-large-32k-online"

  const providerUrl = providerConfig.url
  const providerHeaders = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json"
  }

  let modelTemperature = 0.4
  let selectedModel = isProModel ? proModel : defaultModel
  let rateLimitIdentifier
  if (chatSettings.model === GPT4.modelId) {
    rateLimitIdentifier = "gpt-4"
  } else if (isProModel) {
    rateLimitIdentifier = "hackergpt-pro"
  } else {
    rateLimitIdentifier = "hackergpt"
  }

  let rateLimitCheckResult = await checkRatelimitOnApi(
    profile.user_id,
    rateLimitIdentifier
  )

  let providerRouting
  if (process.env.OPENROUTER_FIRST_PROVIDER) {
    providerRouting = llmConfig.openrouter.providerRouting
  }

  return {
    providerUrl,
    providerHeaders,
    selectedModel,
    rateLimitCheckResult,
    providerRouting,
    isProModel,
    modelTemperature
  }
}
