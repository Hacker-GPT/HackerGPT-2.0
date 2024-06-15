import { WebhookReceiver, RoomServiceClient, Room } from "livekit-server-sdk"

const {
  LIVEKIT_API_KEY: apiKey,
  LIVEKIT_API_SECRET: apiSecret,
  LIVEKIT_URL: livekitUrl
} = process.env

export async function POST(request: Request) {
  if (!apiKey || !apiSecret || !livekitUrl) {
    return new Response(
      "Environment variables LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL must be set",
      { status: 500 }
    )
  }

  try {
    const client = new RoomServiceClient(livekitUrl, apiKey, apiSecret)
    const receiver = new WebhookReceiver(apiKey, apiSecret)
    const body = await request.text()
    const authorizationHeader =
      request.headers.get("authorization") || undefined
    const event = await receiver.receive(body, authorizationHeader)

    switch (event.event) {
      case "participant_left":
        const participantIdentity = event.participant?.identity
        const roomName = event.room?.name

        if (participantIdentity && !participantIdentity.startsWith("agent")) {
          console.log(
            `Participant ${participantIdentity} left room ${roomName}`
          )
          try {
            const rooms = await client.listRooms()
            const roomExists = rooms.some(
              (room: Room) => room.name === roomName
            )

            if (roomExists) {
              await client.deleteRoom(roomName as string)
              console.log(`Room ${roomName} deleted successfully`)
            } else {
              console.log(`Room ${roomName} does not exist`)
            }
          } catch (deleteError) {
            console.error(`Failed to delete room: ${roomName}`, deleteError)
          }
        }
        break
    }

    return new Response(JSON.stringify({ message: "Event processed" }), {
      status: 200
    })
  } catch (error) {
    console.error("Error processing webhook:", error)
    return new Response(JSON.stringify({ error: "Failed to process event" }), {
      status: 500
    })
  }
}
