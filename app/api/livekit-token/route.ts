const { createLivekitToken } = await import(
  "@/components/conversation/create-livekit-token"
)
import { generateRandomAlphanumeric } from "@/lib/utils"

export async function GET(request: Request) {
  const apiKey = process.env.LIVEKIT_API_KEY
  const apiSecret = process.env.LIVEKIT_API_SECRET
  const livekitUrl = process.env.LIVEKIT_URL

  if (!apiKey || !apiSecret || !livekitUrl) {
    throw new Error(
      "Environment variables LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL must be set"
    )
  }

  try {
    const roomName = `room-${generateRandomAlphanumeric(4)}`
    const identity = `identity-${generateRandomAlphanumeric(4)}`

    const token = createLivekitToken(apiKey, apiSecret, identity, roomName)
    return new Response(
      JSON.stringify({
        identity,
        accessToken: token,
        url: livekitUrl
      }),
      {
        headers: { "Content-Type": "application/json" }
      }
    )
  } catch (e) {
    console.error("Error generating token:", e)
    return new Response((e as Error).message, { status: 500 })
  }
}
