"use server"

import { User } from "@supabase/supabase-js"
import Stripe from "stripe"
import { unixToDateString } from "../utils"
import {
  createSupabaseAdminClient,
  createSupabaseAppServerClient
} from "./server-utils"
import { getStripe } from "./stripe"

/**
 * This feature is helpful in some situations.
 *  - In case a subscription could not be registered in the DB due to a temporary trouble with WebHook.
 *  - In case to restore a subscription to an older version of HackerGPT.
 */
export async function restoreSubscription() {
  if (process.env.NEXT_PUBLIC_ENABLE_STRIPE_RESTORE !== "true") {
    throw new Error("Stripe restore is not enabled")
  }

  const supabase = createSupabaseAppServerClient()
  const user = (await supabase.auth.getUser()).data.user
  if (!user) {
    throw new Error("User not found")
  }

  const stripe = getStripe()
  const email = user.email
  if (!email) {
    throw new Error("User email not found")
  }

  const searchResult = await stripe.customers.search({
    query: `email:'${user.email}'`
  })
  const customer = searchResult.data.filter(c => !c.deleted)[0]
  if (!customer) {
    throw new Error("You have no subscription to restore.")
  }

  const subscriptions = await stripe.subscriptions.list({
    customer: customer.id,
    status: "active"
  })

  const restoreProductIds: string[] = (
    process.env.STRIPE_RESTORE_PRODUCT_IDS || ""
  )
    .split(",")
    .map(id => id.trim())
  restoreProductIds.push(process.env.STRIPE_PRODUCT_ID as string)

  let hasRestored = false
  for (const subscription of subscriptions.data) {
    console.log(restoreProductIds)
    const subscribedProductIds = subscription.items.data.map(item => {
      return item.plan.product
    })
    if (subscribedProductIds.length !== 1) {
      throw new Error("Subscription has more than one product")
    }
    const subscribedProductId = subscribedProductIds[0] as string
    if (restoreProductIds.includes(subscribedProductId)) {
      await restoreToDatabase(stripe, user, subscription.id)
      hasRestored = true
      break
    }
  }

  if (!hasRestored) {
    throw new Error("No subscription to restore")
  }
}

async function restoreToDatabase(
  stripe: Stripe,
  user: User,
  subscriptionId: string
) {
  const supabaseAdmin = createSupabaseAdminClient()
  await stripe.subscriptions.update(subscriptionId, {
    cancel_at_period_end: false
  })
  const subscription = await stripe.subscriptions.retrieve(subscriptionId)

  // check if the user has active subscription already
  const { data: subscriptions, error } = await supabaseAdmin
    .from("subscriptions")
    .select("*")
    .eq("status", "active")
    .eq("user_id", user.id)
  if (error) {
    throw new Error(error.message)
  }
  if (subscriptions.length > 0) {
    throw new Error("Subscription already exists")
  }
  if (!subscription.customer || typeof subscription.customer !== "string") {
    throw new Error("invalid customer value")
  }

  // restore subscription
  const result = await supabaseAdmin.from("subscriptions").upsert(
    {
      subscription_id: subscriptionId,
      user_id: user.id,
      customer_id: subscription.customer,
      status: subscription.status,
      start_date: unixToDateString(subscription.start_date),
      cancel_at: unixToDateString(subscription.cancel_at),
      canceled_at: unixToDateString(subscription.canceled_at),
      ended_at: unixToDateString(subscription.ended_at)
    },
    { onConflict: "subscription_id" }
  )
  if (result.error) {
    console.error(result.error)
    throw new Error(result.error.message)
  }
}
