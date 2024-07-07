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
  extractErrorMessage,
  COMMAND_GENERATION_PROMPT,
  CODE_INTERPRETER_TOOLS,
  closeSandbox,
  truncateText
} from "@/lib/tools/code-interpreter-utils"
import OpenAI from "openai"
import { isPremiumUser } from "@/lib/server/subscription-utils"
import { v4 as uuidv4 } from 'uuid';

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

    const sessionID = uuidv4()
    const userID = profile.user_id

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
            for await (const chunk of handleStream(stream, sessionID, userID)) {
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
  sessionID: string,
  userID: string
) {
  let contentBuffer = ""
  let codeBuffer = ""
  let isCollectingCode = false
  let toolCallExecuted = false

  for await (const chunk of stream) {
    const { delta } = chunk.choices[0]

    if (delta.content) {
      contentBuffer += delta.content
      yield contentBuffer
      contentBuffer = ""
    } else if (delta.tool_calls) {
      const toolCall = delta.tool_calls[0]
      if (toolCall.function?.name === "execute_python") {
        if (toolCallExecuted) {
          yield `${JSON.stringify({ type: "error", content: "Multiple code executions in a single request are not yet supported. Please submit your code in a single block." })}\n`
          return
        }
        isCollectingCode = true
        toolCallExecuted = true
      }
      if (isCollectingCode && toolCall.function?.arguments) {
        codeBuffer += toolCall.function.arguments
      }
    }
  }

  if (isCollectingCode && codeBuffer) {
    yield* handlePythonExecution(
      { function: { name: "execute_python", arguments: codeBuffer } },
      sessionID,
      userID
    )
  }
}

async function* handlePythonExecution(
  toolCall: { function: { name: string; arguments: string } },
  sessionID: string,
  userID: string
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
    yield `${JSON.stringify({ type: "code_interpreter_status", status: "running" })}\n`
    const executionResult = await executeCode(sessionID, code, userID)

    // Handle stdout
    if (executionResult.stdout) {
      yield `${JSON.stringify({ type: "code_interpreter_output", content: [{ text: truncateText(executionResult.stdout) }] })}\n`
    }

    // Handle stderr
    if (executionResult.stderr) {
      yield `${JSON.stringify({ type: "code_interpreter_output", content: [{ text: truncateText(executionResult.stderr) }] })}\n`
    }

    // Handle results
    if (executionResult.results && executionResult.results.length > 0) {
      const formattedResults = executionResult.results.map((result: any) => {
        if (typeof result === "string") {
          return { text: truncateText(result) }
        } else if (result && typeof result === "object") {
          if (result.text) {
            return { text: truncateText(result.text) }
          } else {
            return { text: truncateText(JSON.stringify(result, null, 2)) }
          }
        }
        return { text: truncateText(String(result)) }
      })
      yield `${JSON.stringify({ type: "code_interpreter_output", content: formattedResults })}\n`
    }

    // Handle error
    if (executionResult.error) {
      throw new Error(extractErrorMessage(executionResult.error))
    }

    yield `${JSON.stringify({ type: "code_interpreter_status", status: "finished" })}\n`
  } catch (error: any) {
    console.error("Error executing code:", error)
    const errorMessage =
      error instanceof Error ? error.message : extractErrorMessage(error)
    yield `${JSON.stringify({ type: "error", content: errorMessage })}\n`
    yield `${JSON.stringify({ type: "code_interpreter_status", status: "error" })}\n`
  }
}
