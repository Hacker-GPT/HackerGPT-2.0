"use server"

import { User } from "@supabase/supabase-js"
import Stripe from "stripe"
import { unixToDateString } from "../utils"
import {
  createSupabaseAdminClient,
  createSupabaseAppServerClient
} from "./server-utils"
import {
  getActiveSubscriptions,
  getCustomerByEmail,
  getRestoreProductIds,
  getStripe,
  isRestoreableSubscription
} from "./stripe"
import { Tables } from "@/supabase/types"

/**
 * This feature is helpful in some situations.
 *  - In case a subscription could not be registered in the DB due to a temporary trouble with WebHook.
 *  - In case to restore a subscription to an older version of HackerGPT.
 */
export async function restoreSubscription(): Promise<Tables<"subscriptions"> | null> {
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

  const customer = await getCustomerByEmail(stripe, email!)
  if (!customer) {
    return null
  }

  const subscriptions = await getActiveSubscriptions(stripe, customer.id)
  for (const subscription of subscriptions.data) {
    if (isRestoreableSubscription(subscription)) {
      const restoredItem = await restoreToDatabase(
        stripe,
        user,
        subscription.id
      )
      return restoredItem
    }
  }

  throw new Error("No subscription to restore")
}

async function restoreToDatabase(
  stripe: Stripe,
  user: User,
  subscriptionId: string
): Promise<Tables<"subscriptions">> {
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
    return subscriptions[0]
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
  const newSubscription = await supabaseAdmin
    .from("subscriptions")
    .select("*")
    .eq("status", "active")
    .eq("user_id", user.id)
    .single()
  if (newSubscription.error) {
    throw new Error(newSubscription.error.message)
  }
  return newSubscription.data
}
