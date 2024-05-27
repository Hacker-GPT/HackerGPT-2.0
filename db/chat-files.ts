import { supabase } from "@/lib/supabase/browser-client"
import { TablesInsert } from "@/supabase/types"

export const getChatFilesByChatId = async (chatId: string) => {
  const { data: chatFiles, error } = await supabase
    .from("chats")
    .select(
      `
      id, 
      name, 
      files (*)
    `
    )
    .eq("id", chatId)
    .single()

  if (!chatFiles) {
    throw new Error(error.message)
  }

  return chatFiles
}

export const createChatFiles = async (
  chatFiles: TablesInsert<"chat_files">[]
) => {
  const { data: createdChatFiles, error } = await supabase
    .from("chat_files")
    .insert(chatFiles)
    .select("*")

  if (!createdChatFiles) {
    throw new Error(error.message)
  }

  return createdChatFiles
}
