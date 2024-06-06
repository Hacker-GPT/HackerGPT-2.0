import { AccessToken } from "livekit-server-sdk"

export const createLivekitToken = (
  apiKey: string,
  apiSecret: string,
  identity: string,
  roomName: string
) => {
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
