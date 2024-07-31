import { checkRatelimitOnApi } from "@/lib/server/ratelimiter"
import { getAIProfile } from "@/lib/server/server-chat-helpers"
import { NextRequest, NextResponse } from "next/server"

export async function POST(req: NextRequest) {
  try {
    const { text, format = "mp3" }: { text: string; format?: string } =
      await req.json()
    const profile = await getAIProfile()

    const rateLimitCheckResult = await checkRatelimitOnApi(
      profile.user_id,
      "tts-1"
    )

    if (rateLimitCheckResult !== null) {
      return rateLimitCheckResult.response
    }

    const MAX_LENGTH = 4096
    const truncationNotice =
      " Please note, this message was shortened to fit the length limit."
    const maxTextLength = MAX_LENGTH - truncationNotice.length
    const needsTruncation = text.length > MAX_LENGTH
    const truncatedText = needsTruncation
      ? text.slice(0, maxTextLength) + truncationNotice
      : text

    const openaiResponse = await fetch(
      "https://api.openai.com/v1/audio/speech",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${process.env.OPENAI_API_KEY}`
        },
        body: JSON.stringify({
          model: "tts-1",
          voice: "onyx",
          input: truncatedText,
          response_format: format,
          speed: 1,
          stream: true
        })
      }
    )

    if (!openaiResponse.ok) {
      const errorData = await openaiResponse.json()
      throw new Error(
        `HTTP error! status: ${openaiResponse.status}, ${errorData.error.message}`
      )
    }

    if (!openaiResponse.body) {
      throw new Error("Response body is null")
    }

    const stream = new ReadableStream({
      async start(controller) {
        const reader = openaiResponse.body!.getReader()

        async function push() {
          const { done, value } = await reader.read()
          if (done) {
            controller.close()
            return
          }
          controller.enqueue(value)
          await push()
        }

        await push()
      }
    })

    const headers = new Headers()
    headers.append("Content-Type", `audio/${format}`)
    headers.append("Transfer-Encoding", "chunked")

    return new NextResponse(stream, { headers })
  } catch (error: any) {
    console.error("Error generating speech:", {
      message: error.message,
      stack: error.stack,
      status: error.status || 500,
      details: error.response ? await error.response.json() : null
    })
    return new NextResponse("Error generating speech", { status: 500 })
  }
}
