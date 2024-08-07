import { getAIProfile } from "@/lib/server/server-chat-helpers"
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
import { getSelectedModel } from "@/lib/models/notdiamond"

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
  try {
    const { payload, chatImages } = await request.json()
    const profile = await getAIProfile()

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
      null
    )
    updateOrAddSystemMessage(
      cleanedMessages,
      llmConfig.systemPrompts.openaiChat
    )
    filterEmptyAssistantMessages(cleanedMessages)
    replaceWordsInLastUserMessage(cleanedMessages, wordReplacements)

    const selectedModel = await getSelectedModel(cleanedMessages, "openai")

    const requestBody = {
      model: selectedModel,
      messages: cleanedMessages as ChatCompletionCreateParamsBase["messages"],
      temperature: 0.4,
      max_tokens: 1024,
      stream: true
    }

    const res = await fetch(llmConfig.openai.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${llmConfig.openai.apiKey}`
      },
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
