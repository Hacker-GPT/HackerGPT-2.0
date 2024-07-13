import { supabase } from "@/lib/supabase/browser-client"
import { TablesInsert } from "@/supabase/types"

export const createMessageFeedback = async (
  feedback: TablesInsert<"feedback">
) => {
  const { data: createdFeedback, error } = await supabase
    .from("feedback")
    .upsert(feedback, { onConflict: "user_id, chat_id, message_id" })
    .select("*")

  if (!createdFeedback) {
    throw new Error(error.message)
  }

  return createdFeedback
}

export const getFeedbackSummary = async () => {
  const { data: summary, error } = await supabase
    .rpc("get_feedback_summary")
    .select("*")

  if (error) {
    throw new Error(error.message)
  }

  if (!summary || summary.length === 0) {
    throw new Error("No summary found")
  }

  return summary[0]
}

export const deleteFeedbackReview = async (id: string) => {
  const { error } = await supabase
    .from("feedback_reviews")
    .delete()
    .eq("feedback_id", id)

  if (error) {
    throw new Error(error.message)
  }
}

export const upsertFeedbackReview = async (
  feedback_id: string,
  user_id: string,
  notes: string
) => {
  const { data, error } = await supabase
    .from("feedback_reviews")
    .upsert({
      feedback_id,
      reviewed_by: user_id,
      reviewed_at: new Date().toISOString(),
      notes: notes
    })
    .select()

  if (error) {
    throw new Error(error.message)
  }

  return data
}
