import React, { useContext } from "react"
import { ChatbotUIContext } from "@/context/context"
import PluginSelector from "./plugin-selector"

export const EnhancedMenuPicker: React.FC = () => {
  const { setSelectedPluginType } = useContext(ChatbotUIContext)

  const handleSelectPlugin = (type: string) => {
    setSelectedPluginType(type)
  }

  return (
    <>
      <div className="bg-secondary flex flex-col space-y-1 rounded-xl p-2.5 text-sm">
        <PluginSelector onPluginSelect={handleSelectPlugin} />
      </div>
    </>
  )
}
