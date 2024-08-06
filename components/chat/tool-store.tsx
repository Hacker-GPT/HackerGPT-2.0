import { Dialog, Transition, Tab } from "@headlessui/react"
import React, { useState, useEffect, Fragment, useContext, useRef } from "react"
import {
  IconX,
  IconSearch,
  IconCode,
  IconCircleX,
  IconCloudDownload
} from "@tabler/icons-react"

import { PluginSummary } from "@/types/plugins"
import Image from "next/image"
import { PentestGPTContext } from "@/context/context"

interface PluginStorePageProps {
  pluginsData: PluginSummary[]
  installPlugin: any
  uninstallPlugin: any
}

function ToolStorePage({
  pluginsData,
  installPlugin,
  uninstallPlugin
}: PluginStorePageProps) {
  const filters = [
    // "All",
    "Free",
    "Recon tools",
    "Vulnerability scanners",
    "Installed"
  ]
  const [selectedFilter, setSelectedFilter] = useState("All")
  const { isMobile } = useContext(PentestGPTContext)
  const [searchTerm, setSearchTerm] = useState("")
  const categoryRefs = useRef<{ [key: string]: React.RefObject<HTMLDivElement> }>({})

  useEffect(() => {
    filters.forEach(filter => {
      categoryRefs.current[filter] = React.createRef<HTMLDivElement>()
    })
  }, [])

  const scrollToCategory = (category: string) => {
    categoryRefs.current[category]?.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const excludedPluginIds = [0, 99]

  const filteredPlugins = pluginsData
    .filter(plugin => !excludedPluginIds.includes(plugin.id))
    .filter(plugin => {
      const matchesSearch = plugin.name.toLowerCase().includes(searchTerm.toLowerCase())
      return matchesSearch
    })

  const categorizedPlugins = filters.reduce((acc, filter) => {
    acc[filter] = filteredPlugins.filter(plugin => {
      // if (filter === "All") return true
      if (filter === "Installed") return plugin.isInstalled
      if (filter === "Free") return !plugin.isPremium
      if (filter === "Recon tools") return plugin.categories.includes("recon")
      if (filter === "Vulnerability scanners") return plugin.categories.includes("vuln-scanners")
      return false
    })
    return acc
  }, {} as { [key: string]: PluginSummary[] })

    return (
      <div className="h-full overflow-y-auto p-6">
        <div className="mb-6 text-center">
          <h1 className="text-primary mb-4 text-2xl font-bold">Plugin Store</h1>
  
          {/* Category Selection */}
          <Tab.Group>
            <Tab.List className="flex flex-wrap justify-center space-x-2 rounded-xl">
              {filters.map(filter => (
                <Tab
                  key={filter}
                  className={({ selected }: { selected: boolean }) =>
                    `rounded-lg border border-pgpt-light-gray px-3 py-2 text-sm font-medium ${
                      selected
                        ? "bg-muted shadow"
                        : "bg-primary text-secondary hover:bg-primary/[0.40] hover:text-primary"
                    } mb-2`
                  }
                  onClick={() => {
                    setSelectedFilter(filter)
                    scrollToCategory(filter)
                  }}
                >
                  {filter}
                </Tab>
              ))}
            </Tab.List>
          </Tab.Group>
        </div>
  
        {/* Search Bar */}
        <div className="mb-6 flex justify-center">
          <div className="relative w-full max-w-md">
            <div className="pointer-events-none absolute inset-y-0 left-0 flex items-center pl-3">
              <IconSearch className="size-5 text-gray-500" aria-hidden="true" />
            </div>
            <input
              type="search"
              placeholder="Search plugins"
              className="bg-secondary text-primary block w-full rounded-lg py-2 pl-10 pr-3 text-sm focus:border-blue-500 focus:ring-blue-500"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
            />
          </div>
        </div>
  
        {/* Plugin List */}
        {filters.map(filter => (
          <div key={filter} ref={categoryRefs.current[filter]}>
            <h2 className="text-primary mb-4 text-xl font-semibold">{filter}</h2>
            <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-3 mb-8">
              {categorizedPlugins[filter].length > 0 ? (
                categorizedPlugins[filter].map(plugin => (
                  <div
                    key={plugin.id}
                    className="border-pgpt-light-gray flex h-[200px] w-full flex-col justify-between rounded-lg border p-4 shadow"
                  >
                <div className="flex items-center">
                  <div className="mr-4 size-[60px] shrink-0">
                    <Image
                      src={
                        plugin.icon ||
                        "https://avatars.githubusercontent.com/u/148977464"
                      }
                      alt={plugin.name}
                      width={60}
                      height={60}
                      className={`size-full rounded object-cover ${
                        plugin.invertInDarkMode
                          ? "dark:brightness-0 dark:invert"
                          : ""
                      }`}
                    />
                  </div>

                  <div className="flex flex-col justify-between">
                    <h4 className="text-primary flex items-center text-lg">
                      <span className="font-medium">{plugin.name}</span>
                      {plugin.isPremium && (
                        <span className="ml-2 rounded bg-yellow-200 px-2 py-1 text-xs font-semibold uppercase text-yellow-700 shadow">
                          Pro
                        </span>
                      )}
                    </h4>
                    <button
                      className={`mt-2 inline-flex items-center justify-center rounded-lg px-3 py-1.5 text-sm ${
                        plugin.isInstalled
                          ? "bg-red-500 text-white hover:bg-red-600"
                          : "bg-blue-500 text-white hover:bg-blue-600"
                      }`}
                      onClick={() =>
                        plugin.isInstalled
                          ? uninstallPlugin(plugin.id)
                          : installPlugin(plugin.id)
                      }
                    >
                      {plugin.isInstalled ? (
                        <>
                          Uninstall
                          <IconCircleX
                            className="ml-1 size-4"
                            aria-hidden="true"
                          />
                        </>
                      ) : (
                        <>
                          Install
                          <IconCloudDownload
                            className="ml-1 size-4"
                            aria-hidden="true"
                          />
                        </>
                      )}
                    </button>
                  </div>
                </div>
                {/* Description and Premium badge */}
                <p className="text-primary/70 line-clamp-3 h-[60px] text-sm">
                  {plugin.description}
                </p>
                {plugin.githubRepoUrl && (
                  <div className="text-primary/60 h-[14px] text-xs">
                    <a
                      href={plugin.githubRepoUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="flex items-center gap-1.5"
                    >
                      View Source
                      <IconCode className="size-4" aria-hidden="true" />
                    </a>
                  </div>
                )}
              </div>
            ))
            ) : (
              <div className="col-span-full flex flex-col items-center justify-center p-10">
                <p className="text-primary text-lg font-semibold">
                  No plugins found in this category
                </p>
              </div>
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

export default ToolStorePage