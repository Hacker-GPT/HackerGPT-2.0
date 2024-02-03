import { Database } from "@/supabase/types"
import { ChatSettings } from "@/types"
import { createClient } from "@supabase/supabase-js"
import { OpenAIStream, StreamingTextResponse } from "ai"
import { ServerRuntime } from "next"
import OpenAI from "openai"
import { ChatCompletionCreateParamsBase } from "openai/resources/chat/completions.mjs"

export const runtime: ServerRuntime = "edge"

export async function POST(request: Request) {
  const json = await request.json()
  const { chatSettings, messages, customModelId } = json as {
    chatSettings: ChatSettings
    messages: any[]
    customModelId: string
  }

  try {
    const supabaseAdmin = createClient<Database>(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    )

    const { data: customModel, error } = await supabaseAdmin
      .from("models")
      .select("*")
      .eq("id", customModelId)
      .single()

    if (!customModel) {
      throw new Error(error.message)
    }

    const custom = new OpenAI({
      apiKey: customModel.api_key || "",
      baseURL: customModel.base_url
    })

    const response = await custom.chat.completions.create({
      model: chatSettings.model as ChatCompletionCreateParamsBase["model"],
      messages: messages as ChatCompletionCreateParamsBase["messages"],
      temperature: chatSettings.temperature,
      stream: true
    })

    const stream = OpenAIStream(response)

    return new StreamingTextResponse(stream)
  } catch (error: any) {
    let errorMessage = error.message || "An unexpected error occurred"
    const errorCode = error.status || 500

    if (errorMessage.toLowerCase().includes("api key not found")) {
      errorMessage =
        "Custom API Key not found. Please set it in your profile settings."
    } else if (errorMessage.toLowerCase().includes("incorrect api key")) {
      errorMessage =
        "Custom API Key is incorrect. Please fix it in your profile settings."
    }

    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}

export function updateOrAddSystemMessage(
  messages: any[],
  systemMessageContent: any
) {
  const systemInstructions = "User Instructions:\n"
  const existingSystemMessageIndex = messages.findIndex(
    msg => msg.role === "system"
  )

  if (existingSystemMessageIndex !== -1) {
    // Existing system message found
    let existingSystemMessage = messages[existingSystemMessageIndex]
    if (!existingSystemMessage.content.includes(systemInstructions)) {
      // Append new content if "User Instructions:" is not found
      existingSystemMessage.content += `${systemMessageContent}` // Added a newline for separation
    }
    // Move the updated system message to the start
    messages.unshift(messages.splice(existingSystemMessageIndex, 1)[0])
  } else {
    // No system message exists, create a new one
    messages.unshift({
      role: "system",
      content: systemMessageContent
    })
  }
}

export type Role = "assistant" | "user" | "system"

export interface Message {
  role: Role
  content: string
}

export class APIError extends Error {
  code: any
  constructor(message: string | undefined, code: any) {
    super(message)
    this.name = "APIError"
    this.code = code
  }
}
