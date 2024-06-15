import { WebhookReceiver, RoomServiceClient } from 'livekit-server-sdk';

const { LIVEKIT_API_KEY: apiKey, LIVEKIT_API_SECRET: apiSecret, LIVEKIT_URL: livekitUrl } = process.env;

export async function POST(request: Request) {
  if (!apiKey || !apiSecret || !livekitUrl) {
    return new Response(
      "Environment variables LIVEKIT_API_KEY, LIVEKIT_API_SECRET, and LIVEKIT_URL must be set",
      { status: 500 }
    );
  }

  try {
    const client = new RoomServiceClient(livekitUrl, apiKey, apiSecret);
    const receiver = new WebhookReceiver(apiKey, apiSecret);
    const body = await request.text();
    const authorizationHeader = request.headers.get('authorization') || undefined;
    const event = await receiver.receive(body, authorizationHeader);

    console.log('Received event:', event);

    switch (event.event) {
      case 'participant_left':
        console.log(`Participant ${event.participant?.identity} left room ${event.room?.name}`);
        await client.deleteRoom(event.room?.name as string);
        break;
      default:
        console.log(`Unhandled event type: ${event.event}`);
    }

    return new Response(JSON.stringify({ message: 'Event processed' }), { status: 200 });
  } catch (error) {
    console.error('Error processing webhook:', error);
    return new Response(JSON.stringify({ error: 'Failed to process event' }), { status: 500 });
  }
}