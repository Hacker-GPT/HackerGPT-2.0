"use client"

import { PentestGPTContext } from "@/context/context"
import { ChatSettings } from "@/types"
import { FC, useContext } from "react"
import { ModelSelect } from "../models/model-select"

interface ChatSettingsFormProps {
  chatSettings: ChatSettings
  onChangeChatSettings: (value: ChatSettings) => void
}

export const ChatSettingsForm: FC<ChatSettingsFormProps> = ({
  chatSettings,
  onChangeChatSettings
}) => {
  const { profile } = useContext(PentestGPTContext)

  if (!profile) return null

  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <ModelSelect
          selectedModelId={chatSettings.model}
          onSelectModel={model => {
            onChangeChatSettings({ ...chatSettings, model })
          }}
        />
      </div>
    </div>
  )
}
