"use client"

import React, {
  createContext,
  useReducer,
  useContext,
  useEffect,
  Dispatch
} from "react"

import { PluginSummary } from "@/types/plugins"

import { availablePlugins } from "@/components/chat/plugin-store"

enum ActionTypes {
  INSTALL_PLUGIN = "INSTALL_PLUGIN",
  UNINSTALL_PLUGIN = "UNINSTALL_PLUGIN"
}

const initialState = {
  installedPlugins: [] as PluginSummary[]
}

interface PluginContextType {
  state: typeof initialState
  dispatch: Dispatch<any>
  installPlugin: (plugin: PluginSummary) => void
  uninstallPlugin: (pluginId: number) => void
}

const PluginContext = createContext<PluginContextType>({
  state: initialState,
  dispatch: () => null, // Placeholder function
  installPlugin: () => {}, // Placeholder function
  uninstallPlugin: () => {} // Placeholder function
})

const pluginReducer = (
  state: { installedPlugins: any[] },
  action: { type: any; payload: { id: any } }
) => {
  switch (action.type) {
    case ActionTypes.INSTALL_PLUGIN:
      if (!state.installedPlugins.some(p => p.id === action.payload.id)) {
        const updatedPlugins = [...state.installedPlugins, action.payload]
        localStorage.setItem("installedPlugins", JSON.stringify(updatedPlugins))
        return { ...state, installedPlugins: updatedPlugins }
      }
      return state
    case ActionTypes.UNINSTALL_PLUGIN:
      const updatedPlugins = state.installedPlugins.filter(
        p => p.id !== action.payload
      )
      localStorage.setItem("installedPlugins", JSON.stringify(updatedPlugins))
      return { ...state, installedPlugins: updatedPlugins }
    default:
      return state
  }
}

export const PluginProvider = ({ children }: { children: React.ReactNode }) => {
  const [state, dispatch] = useReducer(pluginReducer, initialState)

  // Function to install a plugin
  const installPlugin = (plugin: PluginSummary) => {
    dispatch({ type: ActionTypes.INSTALL_PLUGIN, payload: plugin })
  }

  // Function to uninstall a plugin
  const uninstallPlugin = (pluginId: number) => {
    dispatch({ type: ActionTypes.UNINSTALL_PLUGIN, payload: { id: pluginId } })
  }

  useEffect(() => {
    const localData = localStorage.getItem("installedPlugins")
    let installedPlugins = localData ? JSON.parse(localData) : []

    const defaultPluginIds = [1, 2, 7]

    if (!localData) {
      defaultPluginIds.forEach(id => {
        const defaultPlugin = availablePlugins.find(p => p.id === id)
        if (defaultPlugin) {
          installedPlugins.push({ ...defaultPlugin, isInstalled: true })
        }
      })

      localStorage.setItem("installedPlugins", JSON.stringify(installedPlugins))
    }

    installedPlugins.forEach((plugin: any) => {
      dispatch({ type: ActionTypes.INSTALL_PLUGIN, payload: plugin })
    })
  }, [])

  return (
    <PluginContext.Provider
      value={{ state, dispatch, installPlugin, uninstallPlugin }}
    >
      {children}
    </PluginContext.Provider>
  )
}

export const usePluginContext = () => useContext(PluginContext)
