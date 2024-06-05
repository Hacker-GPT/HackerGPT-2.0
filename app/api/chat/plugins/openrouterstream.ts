import { Message } from "@/types/chat"
import {
  replaceWordsInLastUserMessage,
  wordReplacements
} from "@/lib/ai-helper"
import {
  ParsedEvent,
  ReconnectInterval,
  createParser
} from "eventsource-parser"
import llmConfig from "@/lib/models/llm/llm-config"
import { APIError } from "@/lib/models/llm/api-error"

export interface OpenAIModel {
  id: string
  name: string
  maxLength: number
  tokenLimit: number
}

export const OpenRouterStream = async (
  messages: any[],
  answerMessage: Message,
  tools?: any
) => {
  let providerRouting, model
  const SYSTEM_PROMPT = `${llmConfig.systemPrompts.openaiCurrentDateOnly}. Don't forget to always provide a command inside of the code block.`

  replaceWordsInLastUserMessage(messages, wordReplacements)

  // Check if any message contains an image
  const containsImage = messages.some(
    (message: any) =>
      Array.isArray(message.content) &&
      message.content.some((item: any) => item.type === "image_url")
  )

  const lastMessage = messages[messages.length - 1]
  if (Array.isArray(lastMessage.content)) {
    lastMessage.content = lastMessage.content.map((item: any) =>
      item.type === "text" ? { ...item, text: answerMessage.content } : item
    )
  } else {
    lastMessage.content = answerMessage.content
  }

  // Set model based on whether any message contains an image
  model = containsImage ? "openai/gpt-4o" : "openai/gpt-4o"

  // if (model !== "openai/gpt-4o" && process.env.OPENROUTER_FIRST_PROVIDER) {
  //   providerRouting = llmConfig.openrouter.providerRouting
  // }

  const commonBody = {
    model,
    messages: [
      {
        role: "system",
        content: SYSTEM_PROMPT
      },
      ...messages
    ],
    max_tokens: 1024,
    temperature: 0.1,
    stream: true
    // ...(providerRouting && { provider: providerRouting })
    // ...(tools && Object.keys(tools).length > 0
    //   ? { tools: tools, tool_choice: "auto" }
    //   : {}),
  }

  const requestOptions = {
    method: "POST",
    headers: {
      Authorization: `Bearer ${llmConfig.openrouter.apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify(commonBody)
  }

  const res = await fetch(llmConfig.openrouter.url, requestOptions)

  if (!res.ok) {
    const result = await res.json()
    const errorMessage = result.error?.message || "An unknown error occurred"
    const statusErrorMap: { [key: number]: string } = {
      400: "Bad Request",
      401: "Invalid Credentials",
      402: "Out of Credits",
      403: "Moderation Required",
      408: "Request Timeout",
      429: "Rate Limited",
      502: "Service Unavailable"
    }
    throw new APIError(
      `${statusErrorMap[res.status] || "HTTP Error"}: ${errorMessage}`,
      res.status
    )
  }

  if (!res.body) {
    throw new Error("Response body is null")
  }

  const encoder = new TextEncoder()
  const decoder = new TextDecoder()

  const stream = new ReadableStream({
    async start(controller) {
      const onParse = (event: ParsedEvent | ReconnectInterval) => {
        if (event.type === "event") {
          const data = event.data
          if (data !== "[DONE]") {
            try {
              const json = JSON.parse(data)
              if (json.choices[0].finish_reason != null) {
                controller.close()
                return
              }

              let text = json.choices[0].delta.content

              if (
                tools &&
                json.choices[0].delta.tool_calls &&
                json.choices[0].delta.tool_calls.length > 0
              ) {
                text = json.choices[0].delta.tool_calls[0].function.arguments
              }

              const queue = encoder.encode(text)
              controller.enqueue(queue)
            } catch (e) {
              controller.error(e)
            }
          } else {
            controller.close()
          }
        }
      }

      const parser = createParser(onParse)

      for await (const chunk of res.body as any) {
        const content = decoder.decode(chunk)
        if (content.trim() === "data: [DONE]") {
          controller.close()
        } else {
          parser.feed(content)
        }
      }
    }
  })

  return stream
}
