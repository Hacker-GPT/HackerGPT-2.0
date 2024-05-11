"use server"

import {
  cancelSubscription,
  getActiveSubscriptions,
  getCustomersByEmail,
  getStripe
} from "@/lib/server/stripe"
import { createSupabaseAdminClient } from "@/lib/server/server-utils"
import { errStr } from "@/lib/result"

export const deleteUser = async (userId: string) => {
  const supabase = createSupabaseAdminClient()
  const user = (await supabase.auth.getUser()).data.user
  if (!user) {
    return errStr("User not found")
  }

  const customers = await getCustomersByEmail(getStripe(), user?.email || "")
  if (customers.length !== 0) {
    for (const customer of customers) {
      const subscriptions = await getActiveSubscriptions(
        getStripe(),
        customer.id
      )

      for (const subscription of subscriptions.data) {
        await cancelSubscription(getStripe(), subscription.id)
      }
    }
  }

  const { error: feedbackError } = await supabase
    .from("feedback")
    .delete()
    .eq("user_id", userId)

  if (feedbackError) {
    throw new Error(feedbackError.message)
  }

  const { error: assistantCollectionsError } = await supabase
    .from("assistant_collections")
    .delete()
    .eq("user_id", userId)

  if (assistantCollectionsError) {
    throw new Error(assistantCollectionsError.message)
  }

  const { error: assistantFilesError } = await supabase
    .from("assistant_files")
    .delete()
    .eq("user_id", userId)

  if (assistantFilesError) {
    throw new Error(assistantFilesError.message)
  }

  const { error: assistantToolsError } = await supabase
    .from("assistant_tools")
    .delete()
    .eq("user_id", userId)

  if (assistantToolsError) {
    throw new Error(assistantToolsError.message)
  }

  const { error: assistantWorkspacesError } = await supabase
    .from("assistant_workspaces")
    .delete()
    .eq("user_id", userId)

  if (assistantWorkspacesError) {
    throw new Error(assistantWorkspacesError.message)
  }

  const { error: assistantsError } = await supabase
    .from("assistants")
    .delete()
    .eq("user_id", userId)

  if (assistantsError) {
    throw new Error(assistantsError.message)
  }

  const { error: chatFilesError } = await supabase
    .from("chat_files")
    .delete()
    .eq("user_id", userId)

  if (chatFilesError) {
    throw new Error(chatFilesError.message)
  }

  const { error: chatsError } = await supabase
    .from("chats")
    .delete()
    .eq("user_id", userId)

  if (chatsError) {
    throw new Error(chatsError.message)
  }

  const { error: collectionFilesError } = await supabase
    .from("collection_files")
    .delete()
    .eq("user_id", userId)

  if (collectionFilesError) {
    throw new Error(collectionFilesError.message)
  }

  const { error: collectionWorkspacesError } = await supabase
    .from("collection_workspaces")
    .delete()
    .eq("user_id", userId)

  if (collectionWorkspacesError) {
    throw new Error(collectionWorkspacesError.message)
  }

  const { error: collectionsError } = await supabase
    .from("collections")
    .delete()
    .eq("user_id", userId)

  if (collectionsError) {
    throw new Error(collectionsError.message)
  }

  const { error: fileItemsError } = await supabase
    .from("file_items")
    .delete()
    .eq("user_id", userId)

  if (fileItemsError) {
    throw new Error(fileItemsError.message)
  }

  const { error: fileWorkspacesError } = await supabase
    .from("file_workspaces")
    .delete()
    .eq("user_id", userId)

  if (fileWorkspacesError) {
    throw new Error(fileWorkspacesError.message)
  }

  const { error: filesError } = await supabase
    .from("files")
    .delete()
    .eq("user_id", userId)

  if (filesError) {
    throw new Error(filesError.message)
  }

  const { error: foldersError } = await supabase
    .from("folders")
    .delete()
    .eq("user_id", userId)

  if (foldersError) {
    throw new Error(foldersError.message)
  }

  const { error: messageFileItemsError } = await supabase
    .from("message_file_items")
    .delete()
    .eq("user_id", userId)

  if (messageFileItemsError) {
    throw new Error(messageFileItemsError.message)
  }

  const { error: messagesError } = await supabase
    .from("messages")
    .delete()
    .eq("user_id", userId)

  if (messagesError) {
    throw new Error(messagesError.message)
  }

  const { error: modelWorkspacesError } = await supabase
    .from("model_workspaces")
    .delete()
    .eq("user_id", userId)

  if (modelWorkspacesError) {
    throw new Error(modelWorkspacesError.message)
  }

  const { error: modelsError } = await supabase
    .from("models")
    .delete()
    .eq("user_id", userId)

  if (modelsError) {
    throw new Error(modelsError.message)
  }

  const { error: presetWorkspacesError } = await supabase
    .from("preset_workspaces")
    .delete()
    .eq("user_id", userId)

  if (presetWorkspacesError) {
    throw new Error(presetWorkspacesError.message)
  }

  const { error: presetsError } = await supabase
    .from("presets")
    .delete()
    .eq("user_id", userId)

  if (presetsError) {
    throw new Error(presetsError.message)
  }

  const { error: profilesError } = await supabase
    .from("profiles")
    .delete()
    .eq("user_id", userId)

  if (profilesError) {
    console.log(profilesError)
    throw new Error(profilesError.message)
  }

  const { error: promptWorkspacesError } = await supabase
    .from("prompt_workspaces")
    .delete()
    .eq("user_id", userId)

  if (promptWorkspacesError) {
    throw new Error(promptWorkspacesError.message)
  }

  const { error: promptsError } = await supabase
    .from("prompts")
    .delete()
    .eq("user_id", userId)

  if (promptsError) {
    throw new Error(promptsError.message)
  }

  const { error: subscriptionsError } = await supabase
    .from("subscriptions")
    .delete()
    .eq("user_id", userId)

  if (subscriptionsError) {
    throw new Error(subscriptionsError.message)
  }

  const { error: toolsWorkspaceError } = await supabase
    .from("tool_workspaces")
    .delete()
    .eq("user_id", userId)

  if (toolsWorkspaceError) {
    throw new Error(toolsWorkspaceError.message)
  }

  const { error: toolsError } = await supabase
    .from("tools")
    .delete()
    .eq("user_id", userId)

  if (toolsError) {
    throw new Error(toolsError.message)
  }

  const { error: workspacesError } = await supabase
    .from("workspaces")
    .delete()
    .eq("user_id", userId)

  if (workspacesError) {
    throw new Error(workspacesError.message)
  }

  const { data, error } = await supabase.from("users").delete().eq("id", userId)

  if (error) {
    throw new Error(error?.message)
  }

  const response = await fetch(
    `${process.env.SUPABASE_URL}/auth/admin/users/${userId}`,
    {
      method: "DELETE",
      headers: new Headers({
        apikey: process.env.SUPABASE_SERVICE_KEY || "",
        Authorization: `Bearer ${process.env.SUPABASE_SERVICE_KEY}`,
        "Content-Type": "application/json"
      })
    }
  )

  if (!response.ok) {
    throw new Error(
      `An error occurred when trying to delete your user data. status: ${response.status}`
    )
  }

  await supabase.auth.signOut()

  return true
}
