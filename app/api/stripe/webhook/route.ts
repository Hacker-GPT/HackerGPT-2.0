// Follow this setup guide to integrate the Deno language server with your editor:
// https://deno.land/manual/getting_started/setup_your_environment
// This enables autocomplete, go to definition, etc.

import { createClient } from "@supabase/supabase-js"

// Import via bare specifier thanks to the import_map.json file.
import Stripe from "stripe"

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

const stripe = new Stripe(process.env.STRIPE_API_KEY as string, {
  // This is needed to use the Fetch API rather than relying on the Node http
  // package.
  apiVersion: "2023-10-16",
  httpClient: Stripe.createFetchHttpClient()
})
// This is needed in order to use the Web Crypto API in Deno.
const cryptoProvider = Stripe.createSubtleCryptoProvider()

export const runtime = "edge"

export async function POST(request: Request) {
  const signature = request.headers.get("Stripe-Signature")

  // First step is to verify the event. The .text() method must be used as the
  // verification relies on the raw request body rather than the parsed JSON.
  const body = await request.text()
  let receivedEvent
  try {
    receivedEvent = await stripe.webhooks.constructEventAsync(
      body,
      signature!,
      process.env.STRIPE_WEBHOOK_SIGNING_SECRET!,
      undefined,
      cryptoProvider
    )
  } catch (err: any) {
    console.error(err.message)
    return new Response(err.message, { status: 400 })
  }
  console.log(`🔔 Event received: ${receivedEvent.id} ${receivedEvent.type}`)

  // Reference:
  // https://stripe.com/docs/billing/subscriptions/build-subscriptions
  switch (receivedEvent.type) {
    case "checkout.session.completed":
      // Payment is successful and the subscription is created.
      // You should provision the subscription and save the customer ID to your database.
      await activateSubscription(
        receivedEvent.data.object.subscription as string,
        receivedEvent.data.object.customer as string
      )
      break
    case "invoice.paid":
      // Continue to provision the subscription as payments continue to be made.
      // Store the status in your database and check when a user accesses your service.
      // This approach helps you avoid hitting rate limits.
      await extendSubscription(
        receivedEvent.data.object.subscription as string,
        receivedEvent.data.object.customer as string
      )
      break
    case "invoice.payment_failed":
      // The payment failed or the customer does not have a valid payment method.
      // The subscription becomes past_due. Notify your customer and send them to the
      // customer portal to update their payment information.
      break
    default:
    // Unhandled event type
  }

  return new Response(JSON.stringify({ ok: true }), { status: 200 })
}

async function activateSubscription(
  subscriptionId: string,
  customerId: string
) {
  console.log("activateSubscription", subscriptionId, customerId)
  const profile = await supabase
    .from("profiles")
    .select("*")
    .eq("id", customerId)
  if (!profile.data) {
    throw new Error("No profile found")
  }

  const subscription = await stripe.subscriptions.retrieve(subscriptionId)
  const result = await supabase.from("subscriptions").upsert(
    {
      subscription_id: subscriptionId,
      user_id: customerId,
      status: subscription.status,
      ended_at: new Date()
    },
    { onConflict: "subscription_id" }
  )
  if (result.error) {
    console.error(result.error)
    throw new Error(result.error.message)
  }
}

async function extendSubscription(subscriptionId: string, customerId: string) {
  console.log("extendSubscription", subscriptionId, customerId)
  const profile = await supabase
    .from("profiles")
    .select("*")
    .eq("id", customerId)
  if (!profile.data) {
    throw new Error("No profile found")
  }

  const subscription = await stripe.subscriptions.retrieve(subscriptionId)
  const result = await supabase.from("subscriptions").upsert(
    {
      subscription_id: subscriptionId,
      user_id: customerId,
      status: subscription.status,
      ended_at: subscription.ended_at
    },
    { onConflict: "subscription_id" }
  )
  if (result.error) {
    console.error(result.error)
    throw new Error(result.error.message)
  }
}
