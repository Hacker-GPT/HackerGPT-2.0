import Stripe from "stripe"

export function getStripe(): Stripe {
  const apiKey = process.env.STRIPE_API_KEY
  if (typeof apiKey !== "string") {
    throw new Error("Missing Stripe API key")
  }
  const stripe = new Stripe(apiKey, {
    apiVersion: "2023-10-16"
  })
  return stripe
}
