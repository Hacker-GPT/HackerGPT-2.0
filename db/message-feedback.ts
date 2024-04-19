import { supabase } from "@/lib/supabase/browser-client"
import { TablesInsert } from "@/supabase/types"

export const createMessageFeedback = async (
  feedback: TablesInsert<"feedback">
) => {
  const { data: createdFeedback, error } = await supabase
    .from("feedback")
    .upsert(feedback, { onConflict: "message_id" })
    .select("*")

  if (!createdFeedback) {
    throw new Error(error.message)
  }

  return createdFeedback
}
