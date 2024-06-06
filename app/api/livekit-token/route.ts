import { AccessToken } from "livekit-server-sdk"
import { generateRandomAlphanumeric } from "@/lib/utils"

const apiKey = process.env.LIVEKIT_API_KEY
const apiSecret = process.env.LIVEKIT_API_SECRET
const livekitUrl = process.env.LIVEKIT_URL

if (!apiKey || !apiSecret || !livekitUrl) {
  throw new Error(
    "Environment variables LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL must be set"
  )
}

const createToken = (identity: string, roomName: string) => {
  const at = new AccessToken(apiKey, apiSecret, { identity })
  at.addGrant({
    room: roomName,
    roomJoin: true,
    canPublish: true,
    canPublishData: true,
    canSubscribe: true
  })
  return at.toJwt()
}

export async function GET(request: Request) {
  try {
    const roomName = `room-${generateRandomAlphanumeric(4)}`
    const identity = `identity-${generateRandomAlphanumeric(4)}`

    const token = await createToken(identity, roomName)
    return new Response(
      JSON.stringify({
        identity,
        accessToken: token,
        url: process.env.LIVEKIT_URL
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
