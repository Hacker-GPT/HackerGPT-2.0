import React, { useContext } from "react"
import { ChatbotUIContext } from "@/context/context"
import PluginSelector from "./plugin-selector"

export const EnhancedMenuPicker: React.FC = () => {
  const { selectedPluginType, setSelectedPluginType, isPromptPickerOpen } =
    useContext(ChatbotUIContext)

  const handleSelectPlugin = (type: string) => {
    setSelectedPluginType(type)
  }

  return (
    <>
      {isPromptPickerOpen && (
        <div className="bg-background flex flex-col space-y-1 rounded-xl border-2 p-2 text-sm">
          <PluginSelector
            type={selectedPluginType}
            onPluginSelect={handleSelectPlugin}
          />
        </div>
      )}
    </>
  )
}
