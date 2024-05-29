import { OpenAI } from "openai"
import { checkApiKey, getServerProfile } from "@/lib/server/server-chat-helpers"
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
import llmConfig from "@/lib/models/llm/llm-config"
import { evaluateCode, nonEmpty, executePythonCode } from "@/lib/models/e2b"
import {
  OpenAIStream,
  StreamingTextResponse,
  ToolCallPayload,
  StreamData,
  CreateMessage
} from "ai"

const MODEL_NAME = "gpt-4-turbo"
const E2B_API_KEY = process.env.E2B_API_KEY

if (!E2B_API_KEY) {
  throw new Error("E2B_API_KEY environment variable not found")
}

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || ""
})

export const runtime: ServerRuntime = "edge"

// class APIError extends Error {
//   code: any;
//   constructor(message: string | undefined, code: any) {
//     super(message);
//     this.name = "APIError";
//     this.code = code;
//   }
// }

export async function POST(request: Request) {
  const json = await request.json()
  const { payload, chatImages, selectedPlugin } = json as {
    payload: any
    chatImages: any
    selectedPlugin: any
  }

  const chatSettings = payload.chatSettings

  try {
    const profile = await getServerProfile()

    checkApiKey(profile.openai_api_key, "OpenAI")

    // const openAiUrl = "https://api.openai.com/v1/chat/completions";

    // const openAiHeaders = {
    //   "Content-Type": "application/json",
    //   Authorization: `Bearer ${profile.openai_api_key}`,
    // };

    // rate limit check
    const rateLimitCheckResult = await checkRatelimitOnApi(
      profile.user_id,
      chatSettings.model
    )
    if (rateLimitCheckResult !== null) {
      return rateLimitCheckResult.response
    }

    const cleanedMessages = (await buildFinalMessages(
      payload,
      profile,
      chatImages,
      selectedPlugin
    )) as any[]
    const systemMessageContent = `${llmConfig.systemPrompts.openai}`
    updateOrAddSystemMessage(cleanedMessages, systemMessageContent)
    filterEmptyAssistantMessages(cleanedMessages)
    replaceWordsInLastUserMessage(cleanedMessages, wordReplacements)

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      stream: true,
      messages: cleanedMessages as ChatCompletionCreateParamsBase["messages"],
      temperature: 0.4,
      max_tokens: 1024,
      tools: executePythonCode,
      tool_choice: "auto"
    })

    const data = new StreamData()
    const stream = OpenAIStream(response, {
      experimental_onToolCall: async (
        call: ToolCallPayload,
        appendToolCallMessage
      ) => {
        const newMessages: CreateMessage[] = []

        for (const toolCall of call.tools) {
          if (toolCall.func.name === "execute_python_code") {
            const evaluation = await evaluateCode(
              profile.user_id,
              toolCall.func.arguments.code as string
            )

            data.append({
              messageIdx: cleanedMessages.length,
              function_name: "execute_python_code",
              parameters: {
                code: toolCall.func.arguments.code as string
              },
              tool_call_id: toolCall.id,
              evaluation: {
                stdout: evaluation.stdout,
                stderr: evaluation.stderr,
                ...(evaluation.error && {
                  error: {
                    traceback: evaluation.error.traceback,
                    name: evaluation.error.name,
                    value: evaluation.error.value
                  }
                }),
                results: evaluation.results.map(t =>
                  JSON.parse(JSON.stringify(t))
                )
              }
            })

            const msgs = appendToolCallMessage({
              tool_call_id: toolCall.id,
              function_name: "execute_python_code",
              tool_call_result: {
                stdout: evaluation.stdout,
                stderr: evaluation.stderr,
                ...(evaluation.error && {
                  traceback: evaluation.error.traceback,
                  name: evaluation.error.name,
                  value: evaluation.error.value
                }),
                // Pass only text results to the LLM (to avoid passing encoded media files)
                results: evaluation.results
                  .map(result => result.text)
                  .filter(nonEmpty)
              }
            })

            newMessages.push(...msgs)
          }
        }

        return openai.chat.completions.create({
          messages: [...cleanedMessages, ...newMessages],
          model: MODEL_NAME,
          stream: true,
          tools: executePythonCode,
          tool_choice: "auto"
        })
      },
      onCompletion(completion) {
        console.log("completion", completion)
      },
      onFinal(completion) {
        data.close()
      }
    })

    return new StreamingTextResponse(stream, {}, data)
  } catch (error: any) {
    console.error("Error during API call:", error)
    let errorMessage = error.message || "An unexpected error occurred"
    const errorCode = error.status || 500

    if (errorMessage.toLowerCase().includes("api key not found")) {
      errorMessage =
        "OpenAI API Key not found. Please set it in your profile settings."
    } else if (errorMessage.toLowerCase().includes("incorrect api key")) {
      errorMessage =
        "OpenAI API Key is incorrect. Please fix it in your profile settings."
    }

    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}
