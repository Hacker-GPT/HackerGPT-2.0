import { AccessToken, AccessTokenOptions } from "livekit-server-sdk"
import { generateRandomAlphanumeric } from "@/lib/utils"
import { getServerProfile } from "@/lib/server/server-chat-helpers"

const apiKey = process.env.LIVEKIT_API_KEY
const apiSecret = process.env.LIVEKIT_API_SECRET
const livekitUrl = process.env.LIVEKIT_URL

const createToken = (userInfo: AccessTokenOptions, roomName: string) => {
  const at = new AccessToken(apiKey, apiSecret, {
    identity: userInfo.identity,
    ttl: "1m"
  })
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
    if (!apiKey || !apiSecret || !livekitUrl) {
      return new Response(
        "Environment variables LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL must be set",
        { status: 500 }
      )
    }

    const profile = await getServerProfile()

    const roomName = `room-${generateRandomAlphanumeric(4)}-${generateRandomAlphanumeric(4)}`
    const identity = profile.user_id

    const token = await createToken({ identity }, roomName)
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