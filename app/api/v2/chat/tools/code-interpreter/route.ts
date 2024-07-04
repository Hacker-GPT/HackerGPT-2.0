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
import { handleOpenAIApiError } from "@/lib/models/llm/api-error"
import llmConfig from "@/lib/models/llm/llm-config"
import {
  executeCode,
  SYSTEM_PROMPT,
  CODE_INTERPRETER_TOOLS
} from "@/lib/tools/code-interpreter-utils"

export const runtime: ServerRuntime = "edge"

export async function POST(request: Request) {
  try {
    const { payload, chatImages } = await request.json()
    const profile = await getServerProfile()
    const sessionID = profile.user_id

    const rateLimitCheckResult = await checkRatelimitOnApi(
      profile.user_id,
      "gpt-4"
    )
    if (rateLimitCheckResult !== null) return rateLimitCheckResult.response

    const cleanedMessages = await buildFinalMessages(
      payload,
      profile,
      chatImages,
      null
    )
    updateOrAddSystemMessage(cleanedMessages, SYSTEM_PROMPT)
    filterEmptyAssistantMessages(cleanedMessages)
    replaceWordsInLastUserMessage(cleanedMessages, wordReplacements)

    const res = await fetch(llmConfig.openai.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${llmConfig.openai.apiKey}`
      },
      body: JSON.stringify({
        model: "gpt-4o",
        messages: cleanedMessages as ChatCompletionCreateParamsBase["messages"],
        temperature: 0.4,
        max_tokens: 1024,
        stream: true,
        tools: CODE_INTERPRETER_TOOLS,
        tool_choice: "auto"
      })
    })

    if (!res.ok) {
      await handleOpenAIApiError(res)
    }


    const stream = new ReadableStream({
        async start(controller) {
          const reader = res.body!.getReader()
          const decoder = new TextDecoder()
          let buffer = ""
          let partialToolCall: any = null
  
          while (true) {
            const { done, value } = await reader.read()
            if (done) break
  
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split("\n")
            buffer = lines.pop() || ""
  
            for (const line of lines) {
              if (line.startsWith("data: ")) {
                const data = line.slice(6)
                if (data === "[DONE]") {
                  controller.close()
                  return
                }
                try {
                  const parsed = JSON.parse(data)
                  const choice = parsed.choices[0]
  
                  if (choice.delta.content) {
                    controller.enqueue(choice.delta.content)
                  } else if (choice.delta.tool_calls) {
                  const toolCall = choice.delta.tool_calls[0]
                  if (!partialToolCall)
                    partialToolCall = { function: { name: "", arguments: "" } }
                  partialToolCall.function.name =
                    toolCall.function.name || partialToolCall.function.name
                  partialToolCall.function.arguments +=
                    toolCall.function.arguments || ""

                    if (isCompleteJsonObject(partialToolCall.function.arguments)) {
                        if (partialToolCall.function.name === "execute_python") {
                          try {
                            const parsedArgs = JSON.parse(partialToolCall.function.arguments)
                            const results = await executeCode(sessionID, parsedArgs.code)
                            controller.enqueue(`\n[Code Interpreter Results: ${JSON.stringify(results)}]\n`)
                            
                            // Request AI explanation
                            const explanation = await getAIExplanation(results, parsedArgs.code)
                            controller.enqueue(`\n[AI Explanation: ${explanation}]\n`)
                          } catch (error: any) {
                            console.error("Error executing code:", error)
                            controller.enqueue(`\n[Code Execution Error: ${error.message}]\n`)
                          }
                        }
                        partialToolCall = null
                      }
                    }
                  } catch (error) {
                    console.error("Error parsing JSON:", error, "Raw data:", data)
                  }
                }
              }
            }
          }
        })

    return new Response(stream, {
        headers: { "Content-Type": "text/plain" }
      })
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

async function getAIExplanation(results: any[], code: string): Promise<string> {
    const explanationPrompt = `
      Code executed:
      ${code}
  
      Results:
      ${JSON.stringify(results, null, 2)}
  
      Please provide a concise explanation of the code's output, including any insights or interpretations. If applicable, suggest improvements or alternative approaches.
    `
  
    const response = await fetch(llmConfig.openai.url, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${llmConfig.openai.apiKey}`
      },
      body: JSON.stringify({
        model: "gpt-4",
        messages: [{ role: "user", content: explanationPrompt }],
        temperature: 0.4,
        max_tokens: 300
      })
    })
  
    if (!response.ok) {
      throw new Error("Failed to get AI explanation")
    }
  
    const data = await response.json()
    return data.choices[0].message.content
  }
  
function isCompleteJsonObject(str: string): boolean {
  try {
    JSON.parse(str)
    return true
  } catch {
    return false
  }
}
