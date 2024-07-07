import { ServerRuntime } from "next"
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions.mjs"
import { getServerProfile } from "@/lib/server/server-chat-helpers"
import { checkRatelimitOnApi } from "@/lib/server/ratelimiter"
import {
  buildFinalMessages,
  filterEmptyAssistantMessages
} from "@/lib/build-prompt"
import {
  replaceWordsInLastUserMessage,
  updateOrAddSystemMessage,
  wordReplacements
} from "@/lib/ai-helper"
import llmConfig from "@/lib/models/llm/llm-config"
import {
  executeCode,
  COMMAND_GENERATION_PROMPT,
  CODE_INTERPRETER_TOOLS,
  closeSandbox
} from "@/lib/tools/code-interpreter-utils"
import OpenAI from "openai"
import endent from "endent"
import { isPremiumUser } from "@/lib/server/subscription-utils"

export const runtime: ServerRuntime = "edge"

const openai = new OpenAI({
  apiKey: llmConfig.openai.apiKey
})

export async function POST(request: Request) {
  try {
    const { payload, chatImages, selectedPlugin } = await request.json()
    const profile = await getServerProfile()
    const isPremium = await isPremiumUser(profile.user_id)

    if (!isPremium) {
      return new Response(
        "Access Denied to " +
          "Code Interpreter" +
          ": The tool you are trying to use is exclusive to Pro members. Please upgrade to a Pro account to access this tool."
      )
    }

    const sessionID = profile.user_id

    const rateLimitCheckResult = await checkRatelimitOnApi(
      profile.user_id,
      "gpt-4"
    )
    if (rateLimitCheckResult) return rateLimitCheckResult.response

    const cleanedMessages = await buildFinalMessages(
      payload,
      profile,
      chatImages,
      selectedPlugin
    )
    updateOrAddSystemMessage(cleanedMessages, COMMAND_GENERATION_PROMPT)
    filterEmptyAssistantMessages(cleanedMessages)
    replaceWordsInLastUserMessage(cleanedMessages, wordReplacements)
    const lastUserMessage = cleanedMessages[cleanedMessages.length - 1]
      .content as string

    const stream = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: cleanedMessages as ChatCompletionCreateParamsBase["messages"],
      temperature: 0.4,
      max_tokens: 1024,
      stream: true,
      tools: CODE_INTERPRETER_TOOLS,
      tool_choice: "auto"
    })

    return new Response(
      new ReadableStream({
        async start(controller) {
          try {
            for await (const chunk of handleStream(stream, sessionID)) {
              controller.enqueue(new TextEncoder().encode(chunk))
            }
          } catch (error) {
            console.error("Error in stream processing:", error)
            controller.enqueue(
              new TextEncoder().encode(
                JSON.stringify({ error: "Stream processing error" })
              )
            )
          } finally {
            controller.close()
            await closeSandbox(sessionID)
          }
        }
      }),
      {
        headers: { "Content-Type": "text/plain" }
      }
    )
  } catch (error: any) {
    console.error("Error in API route:", error)
    return new Response(
      JSON.stringify({ error: "An error occurred", details: error.message }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" }
      }
    )
  }
}

async function* handleStream(
  stream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>,
  sessionID: string
) {
  let contentBuffer = ""
  let codeBuffer = ""
  let isCollectingCode = false

  for await (const chunk of stream) {
    const { delta } = chunk.choices[0]

    if (delta.content) {
      contentBuffer += delta.content
      yield contentBuffer
      contentBuffer = ""
    } else if (delta.tool_calls) {
      const toolCall = delta.tool_calls[0]
      if (toolCall.function?.name === "execute_python") {
        isCollectingCode = true
      }
      if (isCollectingCode && toolCall.function?.arguments) {
        codeBuffer += toolCall.function.arguments
      }
    }
  }

  if (isCollectingCode && codeBuffer) {
    yield* handlePythonExecution(
      { function: { name: "execute_python", arguments: codeBuffer } },
      sessionID
    )
  }
}

async function* handlePythonExecution(
  toolCall: { function: { name: string; arguments: string } },
  sessionID: string
) {
  try {
    let code = toolCall.function.arguments

    try {
      const jsonCode = JSON.parse(code)
      if (jsonCode && typeof jsonCode.code === "string") {
        code = jsonCode.code
      }
    } catch (e) {
      // If JSON.parse fails, assume it's raw Python code
    }

    yield `\n${JSON.stringify({ type: "code_interpreter_input", content: code })}\n`

    const results = await executeCode(sessionID, code)
    yield `${JSON.stringify({ type: "code_interpreter_output", content: results.map(text => ({ text })) })}\n`
  } catch (error: any) {
    console.error("Error executing code:", error)
    yield JSON.stringify({ type: "error", content: error.message }) + "\n"
  }
}
