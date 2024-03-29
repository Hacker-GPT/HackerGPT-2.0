import { ChatbotUIContext } from "@/context/context"
import { availablePlugins } from "@/lib/plugins/available-plugins"
import { ChatStarter, PluginID } from "@/types/plugins"
import React, { useContext, useEffect, useState } from "react"
import { useChatHandler } from "./chat-hooks/use-chat-handler"

interface InfoCardProps {
  title: string
  description: string
}

const InfoCard: React.FC<{
  title: string
  description: string
  onClick: () => void
}> = ({ title, description, onClick }) => {
  return (
    <button
      className="hover:bg-secondary rounded-xl border-2 p-3 text-left focus:outline-none"
      onClick={onClick}
    >
      <div className="pb-1 text-sm font-bold">{title}</div>
      <div className="text-xs opacity-75">{description}</div>
    </button>
  )
}

const ChatStarters: React.FC = () => {
  const { selectedPlugin, chatMessages } = useContext(ChatbotUIContext)
  const { handleSendMessage } = useChatHandler()
  const [starters, setStarters] = useState<ChatStarter[]>([])

  useEffect(() => {
    const pluginStarters = availablePlugins.find(
      (plugin: { value: PluginID }) => plugin.value === selectedPlugin
    )?.starters
    setStarters(pluginStarters || [])
  }, [selectedPlugin])

  return (
    <div className="flex items-center justify-start space-x-4">
      <div className="grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
        {starters.map((starter: ChatStarter) => (
          <InfoCard
            title={starter.title}
            description={starter.description}
            key={starter.title}
            onClick={() =>
              handleSendMessage(starter.chatMessage, chatMessages, false)
            }
          />
        ))}
      </div>
    </div>
  )
}

export default ChatStarters
