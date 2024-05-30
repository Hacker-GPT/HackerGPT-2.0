import { ContentType } from "@/types"
import { IconFile, IconMessage, IconPuzzle } from "@tabler/icons-react"
import React, { FC, useContext, useState } from "react"
import { TabsList } from "../ui/tabs"
import { SidebarSwitchItem } from "./sidebar-switch-item"
import { ChatbotUIContext } from "@/context/context"
import PluginStoreModal from "@/components/chat/plugin-store"
import { PluginSummary } from "@/types/plugins"
import { availablePlugins } from "@/lib/plugins/available-plugins"
import { usePluginContext } from "@/components/chat/chat-hooks/PluginProvider"

export const SIDEBAR_ICON_SIZE = 28

interface SidebarSwitcherProps {
  onContentTypeChange: (contentType: ContentType) => void
}

export const SidebarSwitcher: FC<SidebarSwitcherProps> = ({
  onContentTypeChange
}) => {
  const { subscription } = useContext(ChatbotUIContext)
  const [isPluginStoreModalOpen, setIsPluginStoreModalOpen] = useState(false)
  const { state: pluginState, dispatch: pluginDispatch } = usePluginContext()

  const installPlugin = (plugin: PluginSummary) => {
    pluginDispatch({
      type: "INSTALL_PLUGIN",
      payload: { ...plugin, isInstalled: true }
    })
  }

  const uninstallPlugin = (pluginId: number) => {
    pluginDispatch({
      type: "UNINSTALL_PLUGIN",
      payload: pluginId
    })
  }

  const updatedAvailablePlugins = availablePlugins.map(plugin => {
    const isInstalled = pluginState.installedPlugins.some(
      p => p.id === plugin.id
    )
    return { ...plugin, isInstalled }
  })

  return (
    <div className="flex flex-col justify-between">
      <PluginStoreModal
        isOpen={isPluginStoreModalOpen}
        setIsOpen={setIsPluginStoreModalOpen}
        pluginsData={updatedAvailablePlugins}
        installPlugin={installPlugin}
        uninstallPlugin={uninstallPlugin}
      />

      <TabsList className="bg-background flex items-center px-4">
        <SidebarSwitchItem
          icon={<IconMessage size={SIDEBAR_ICON_SIZE} />}
          contentType="chats"
          onContentTypeChange={onContentTypeChange}
        />

        {subscription && (
          <SidebarSwitchItem
            icon={<IconFile size={SIDEBAR_ICON_SIZE} />}
            contentType="files"
            onContentTypeChange={onContentTypeChange}
          />
        )}

        {/* Imitating SidebarSwitchItem but without contentType */}
        <button
          onClick={() => setIsPluginStoreModalOpen(!isPluginStoreModalOpen)}
          className={
            "ring-offset-background focus-visible:ring-ring data-[state=active]:bg-background data-[state=active]:text-foreground inline-flex items-center justify-center whitespace-nowrap rounded-sm px-3 py-1.5 text-sm font-medium transition-all hover:opacity-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 data-[state=active]:shadow-sm"
          }
        >
          <IconPuzzle size={SIDEBAR_ICON_SIZE} />
        </button>
      </TabsList>
    </div>
  )
}
