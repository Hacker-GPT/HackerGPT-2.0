import { streamText } from "ai"
import { createOpenAI as createOpenRouterClient } from "@ai-sdk/openai"
import { Message } from "@/types/chat"
import {
  replaceWordsInLastUserMessage,
  wordReplacements
} from "@/lib/ai-helper"
import llmConfig from "@/lib/models/llm/llm-config"
import { APIError } from "@/lib/models/llm/api-error"

export const OpenRouterStream = async (
  messages: any[],
  answerMessage: Message
) => {
  const SYSTEM_PROMPT = `${llmConfig.systemPrompts.openaiCurrentDateOnly}. Don't forget to always provide a command inside of the code block.`

  replaceWordsInLastUserMessage(messages, wordReplacements)

  const lastMessage = messages[messages.length - 1]
  if (Array.isArray(lastMessage.content)) {
    lastMessage.content = lastMessage.content.map((item: any) =>
      item.type === "text" ? { ...item, text: answerMessage.content } : item
    )
  } else {
    lastMessage.content = answerMessage.content
  }

  const openrouter = createOpenRouterClient({
    baseUrl: "https://openrouter.ai/api/v1",
    apiKey: llmConfig.openrouter.apiKey,
    headers: {
      "HTTP-Referer": "https://pentestgpt.com/plugins",
      "X-Title": "plugins"
    }
  })

  try {
    const result = await streamText({
      model: openrouter("openai/gpt-4o"),
      system: SYSTEM_PROMPT,
      messages: [...messages],
      maxTokens: 512,
      temperature: 0.1
    })

    const encoder = new TextEncoder()

    const stream = new ReadableStream({
      async start(controller) {
        for await (const chunk of result.textStream) {
          const queue = encoder.encode(chunk)
          controller.enqueue(queue)
        }
        controller.close()
      }
    })

    return stream
  } catch (error) {
    console.error("OpenRouterStream error:", error)
    const errorMessage =
      error instanceof Error ? error.message : "An unknown error occurred"
    const statusCode = getErrorStatusCode(errorMessage)
    throw new APIError(`OpenRouter Error: ${errorMessage}`, statusCode)
  }
}

function getErrorStatusCode(errorMessage: string): number {
  const statusErrorMap: { [key: string]: number } = {
    "Bad Request": 400,
    "Invalid Credentials": 401,
    "Out of Credits": 402,
    "Moderation Required": 403,
    "Request Timeout": 408,
    "Rate Limited": 429,
    "Service Unavailable": 502
  }

  for (const [key, value] of Object.entries(statusErrorMap)) {
    if (errorMessage.includes(key)) {
      return value
    }
  }

  return 500
}
