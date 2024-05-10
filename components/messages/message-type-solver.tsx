import { Tables } from "@/supabase/types"
import { FC } from "react"
import { MessageTooLong } from "./message-too-long"
import { MessagePluginFile } from "./message-plugin-file"
import { MessageMarkdown } from "./message-markdown"

interface MessageTypeResolverProps {
  message: Tables<"messages">
  previousMessage: Tables<"messages"> | undefined
  messageSizeLimit: number
  isLastMessage: boolean
}

export const MessageTypeResolver: FC<MessageTypeResolverProps> = ({
  previousMessage,
  message,
  messageSizeLimit,
  isLastMessage
}) => {
  if (
    message.plugin !== null &&
    message.role === "assistant" &&
    previousMessage?.content.startsWith("/") &&
    previousMessage?.content.includes(" -output ")
  ) {
    const outputFilename = previousMessage?.content
    .match(/-output (\S+)/)?.[1]
    ?.trim()
    
    return (
      <MessagePluginFile
        created_at={message.created_at}
        content={message.content}
        plugin={message.plugin}
        id={message.id}
        filename={outputFilename}
        isLastMessage={isLastMessage}
      />
    )
  }

  if (
    message.plugin !== null &&
    message.role === "assistant" &&
    message.content.split("/n/n")[0].includes(" -output ")
  ) {
    const outputFilename = message.content
      .split("/n/n")[0]
      .match(/-output (\S+)/)?.[1]
      ?.trim()

    return (
      <MessagePluginFile
        created_at={message.created_at}
        content={message.content}
        plugin={message.plugin}
        id={message.id}
        filename={outputFilename}
        isLastMessage={isLastMessage}
      />
    )
  }

  if (
    message.plugin !== null &&
    message.role === "assistant" &&
    message.content.length > messageSizeLimit
  ) {
    return (
      <MessageTooLong
        content={message.content}
        plugin={message.plugin}
        id={message.id}
      />
    )
  }

  return <MessageMarkdown content={message.content} />
}
