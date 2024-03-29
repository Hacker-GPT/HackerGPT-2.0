"use client"

import { ChatHelp } from "@/components/chat/chat-help"
import { useChatHandler } from "@/components/chat/chat-hooks/use-chat-handler"
import { ChatInput } from "@/components/chat/chat-input"
import { ChatSettings } from "@/components/chat/chat-settings"
import ChatStarters from "@/components/chat/chat-starters"
import { ChatUI } from "@/components/chat/chat-ui"
import { QuickSettings } from "@/components/chat/quick-settings"
import { Brand, BrandSmall } from "@/components/ui/brand"
import { ChatbotUIContext } from "@/context/context"
import useHotkey from "@/lib/hooks/use-hotkey"
import { useTheme } from "next-themes"
import { useContext } from "react"

export default function ChatPage() {
  useHotkey("o", () => handleNewChat())
  useHotkey("l", () => {
    handleFocusChatInput()
  })

  const { chatMessages, selectedPlugin } = useContext(ChatbotUIContext)

  const { handleNewChat, handleFocusChatInput } = useChatHandler()

  const { theme } = useTheme()

  return (
    <>
      {chatMessages.length === 0 ? (
        <div className="relative flex h-full flex-col items-center justify-center">
          <div className="absolute left-1/2 -translate-x-1/2 -translate-y-3/4">
            <div className="h-sm:hidden mb-12">
              <BrandSmall theme={theme === "dark" ? "dark" : "light"} />
            </div>

            <div className="h-sm:block hidden">
              <Brand theme={theme === "dark" ? "dark" : "light"} />
            </div>
          </div>

          {/* <div className="absolute left-2 top-2">
            <QuickSettings />
          </div> */}

          <div className="absolute top-2 md:left-2">
            <ChatSettings />
          </div>

          <div className="flex grow flex-col items-center justify-center" />

          <div className="z-10 w-full min-w-[300px] items-end px-2 pt-2 sm:w-[600px] md:w-[650px] lg:w-[650px] xl:w-[800px]">
            <ChatStarters
              selectedPlugin={selectedPlugin}
              chatMessages={chatMessages}
            />
          </div>

          <div className="z-10 w-full min-w-[300px] items-end px-2 pb-3 pt-2 sm:w-[600px] sm:pb-8 md:w-[650px] lg:w-[650px] xl:w-[800px]">
            <ChatInput />
          </div>

          <div className="absolute bottom-2 right-2 hidden md:block lg:bottom-4 lg:right-4">
            <ChatHelp />
          </div>
        </div>
      ) : (
        <ChatUI />
      )}
    </>
  )
}
