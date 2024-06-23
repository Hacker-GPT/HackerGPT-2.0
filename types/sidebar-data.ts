import { Tables } from "@/supabase/types"

export type DataListType =
  | Tables<"chats">[]
  | Tables<"files">[]
  | Tables<"tools">[]
  | Tables<"models">[]

export type DataItemType =
  | Tables<"chats">
  | Tables<"files">
  | Tables<"tools">
  | Tables<"models">
