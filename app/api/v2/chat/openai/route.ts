import { getServerProfile } from "@/lib/server/server-chat-helpers"
import { ServerRuntime } from "next"
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions.mjs"

import {
  replaceWordsInLastUserMessage,
  updateOrAddSystemMessage,
  wordReplacements
} from "@/lib/ai-helper"

import { checkRatelimitOnApi } from "@/lib/server/ratelimiter"
import {
  buildFinalMessages,
  filterEmptyAssistantMessages
} from "@/lib/build-prompt"
import { handleOpenAIApiError } from "@/lib/models/llm/api-error"
import llmConfig from "@/lib/models/llm/llm-config"

export const runtime: ServerRuntime = "edge"

const openAiUrl = "https://api.openai.com/v1/chat/completions"
const openAiHeaders = {
  "Content-Type": "application/json",
  Authorization: `Bearer ${llmConfig.openai.apiKey}`
}

export async function POST(request: Request) {
  try {
    const { payload, chatImages, selectedPlugin } = await request.json()
    const profile = await getServerProfile()

    const rateLimitCheckResult = await checkRatelimitOnApi(
      profile.user_id,
      "gpt-4"
    )
    if (rateLimitCheckResult !== null) {
      return rateLimitCheckResult.response
    }

    const cleanedMessages = await buildFinalMessages(
      payload,
      profile,
      chatImages,
      selectedPlugin
    )
    updateOrAddSystemMessage(cleanedMessages, llmConfig.systemPrompts.openai)
    filterEmptyAssistantMessages(cleanedMessages)
    replaceWordsInLastUserMessage(cleanedMessages, wordReplacements)

    const requestBody = {
      model: "gpt-4o",
      messages: cleanedMessages as ChatCompletionCreateParamsBase["messages"],
      temperature: 0.4,
      max_tokens: 1024,
      stream: true
    }

    const res = await fetch(openAiUrl, {
      method: "POST",
      headers: openAiHeaders,
      body: JSON.stringify(requestBody)
    })

    if (!res.ok) {
      await handleOpenAIApiError(res)
    }

    if (!res.body) {
      throw new Error("Response body is null")
    }

    return res
  } catch (error: any) {
    const errorMessage = error.message || "An unexpected error occurred"
    const errorCode = error.status || 500

    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}
