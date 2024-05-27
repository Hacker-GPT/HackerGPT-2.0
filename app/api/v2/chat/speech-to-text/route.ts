import { NextRequest, NextResponse } from "next/server"
import { getServerProfile } from "@/lib/server/server-chat-helpers"
import { checkRatelimitOnApi } from "@/lib/server/ratelimiter"
import llmConfig from "@/lib/models/llm/llm-config"

export async function POST(req: NextRequest) {
  const formData = await req.formData()
  const audioFile = formData.get("audioFile")
  if (!audioFile || !(audioFile instanceof Blob)) {
    return new NextResponse(
      "No audio file provided or invalid file type. Please provide a valid audio file.",
      {
        status: 400
      }
    )
  }

  const profile = await getServerProfile()
  const rateLimitCheckResult = await checkRatelimitOnApi(
    profile.user_id,
    "stt-1"
  )

  if (rateLimitCheckResult !== null) {
    return rateLimitCheckResult.response
  }

  const OPENAI_API_URL = "https://api.openai.com/v1/audio/transcriptions"
  const WHISPER_MODEL = "whisper-1"

  try {
    const buffer = Buffer.from(await audioFile.arrayBuffer())

    const openaiFormData = new FormData()
    openaiFormData.append(
      "file",
      new Blob([buffer], { type: audioFile.type }),
      `audio.${audioFile.type.split("/")[1]}`
    )
    openaiFormData.append("model", WHISPER_MODEL)
    openaiFormData.append("response_format", "text")

    const response = await fetch(OPENAI_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${llmConfig.openai.apiKey}`
      },
      body: openaiFormData
    })

    if (!response.ok) {
      const errorText = await response.text()
      console.error(
        `Failed to transcribe audio: ${response.statusText}`,
        errorText
      )
      throw new Error(`Failed to transcribe audio: ${response.statusText}`)
    }

    const contentType = response.headers.get("content-type")
    let transcription
    if (contentType && contentType.includes("application/json")) {
      transcription = await response.json()
    } else {
      transcription = { text: await response.text() }
    }

    const trimmedText = transcription.text.trim()

    return new NextResponse(JSON.stringify({ text: trimmedText }), {
      status: 200,
      headers: { "Content-Type": "application/json" }
    })
  } catch (error) {
    console.error("Error transcribing audio:", error)
    return new NextResponse("Error transcribing audio", { status: 500 })
  }
}
