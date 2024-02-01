// Follow this setup guide to integrate the Deno language server with your editor:
// https://deno.land/manual/getting_started/setup_your_environment
// This enables autocomplete, go to definition, etc.

// Import via bare specifier thanks to the import_map.json file.
import Stripe from "https://esm.sh/stripe@14.14.0?target=deno"

const stripe = new Stripe(Deno.env.get("STRIPE_API_KEY") as string, {
  // This is needed to use the Fetch API rather than relying on the Node http
  // package.
  apiVersion: "2022-11-15",
  httpClient: Stripe.createFetchHttpClient()
})
// This is needed in order to use the Web Crypto API in Deno.
const cryptoProvider = Stripe.createSubtleCryptoProvider()

console.log("Hello from Stripe Webhook!")

Deno.serve(async request => {
  const signature = request.headers.get("Stripe-Signature")

  // First step is to verify the event. The .text() method must be used as the
  // verification relies on the raw request body rather than the parsed JSON.
  const body = await request.text()
  let receivedEvent
  try {
    receivedEvent = await stripe.webhooks.constructEventAsync(
      body,
      signature!,
      Deno.env.get("STRIPE_WEBHOOK_SIGNING_SECRET")!,
      undefined,
      cryptoProvider
    )
  } catch (err) {
    return new Response(err.message, { status: 400 })
  }
  console.log(`🔔 Event received: ${receivedEvent.id}`)

  // TODO: update the customer's subscription status in the database.
  await updateSubscriptionStatus(
    receivedEvent.data.object.id,
    receivedEvent.data.object.customer
  )

  return new Response(JSON.stringify({ ok: true }), { status: 200 })
})

async function updateSubscriptionStatus(
  subscriptionId: string,
  customerId: string
) {
  const customer = await stripe.customers.retrieve(customerId)
  customer.subscriptions.data.forEach(subscription => {
    if (subscription.id === subscriptionId) {
      if (subscription.status === "active") {
        // update subscription table on supabase postgres
        // with status active
        // with price id
        // with next billing date
        // with customer id
      }
    }
  })
}
