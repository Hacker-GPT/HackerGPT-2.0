import { Tables } from "@/supabase/types"

export type DataListType =
  | Tables<"chats">[]
  | Tables<"files">[]
  | Tables<"tools">[]

export type DataItemType = Tables<"chats"> | Tables<"files"> | Tables<"tools">
