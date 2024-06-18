import { RoomServiceClient } from "livekit-server-sdk"

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

  const client = new RoomServiceClient(livekitUrl, apiKey, apiSecret)

  try {
    const rooms = await client.listRooms()
    const now = Date.now()

    for (const room of rooms) {
      const creationTime = Number(room.creationTime) * 1000 // Convert to milliseconds
      const elapsedTime = now - creationTime

      if (elapsedTime > 15 * 60 * 1000) {
        // 15 minutes in milliseconds
        await client.deleteRoom(room.name)
        console.log(`Room ${room.name} deleted successfully`)
      }
    }

    return new Response(JSON.stringify({ message: "Room check completed" }), {
      status: 200
    })
  } catch (error) {
    console.error("Error checking rooms:", error)
    return new Response(JSON.stringify({ error: "Failed to check rooms" }), {
      status: 500
    })
  }
}
