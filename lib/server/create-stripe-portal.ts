"use server"

export async function getStripePortal(userId: string) {
  const apiKey = process.env.STRIPE_API_KEY
  const stripe = require("stripe")(apiKey)
  let customer = null

  // check if customer exists, if not create them
  try {
    customer = await stripe.customers.retrieve(userId)
  } catch (error: any) {
    if (error.message === `No such customer: '${userId}'`) {
      // no customer found
      customer = await stripe.customers.create({
        id: userId
      })
    } else {
      throw error
    }
  }

  const session = await stripe.billingPortal.sessions.create({
    customer: customer.id,
    return_url: process.env.STRIPE_RETURN_URL
  })
  return session.url
}
