export interface ChatStarter {
  title: string
  description: string
  chatMessage: string
}

export interface PluginSummary {
  id: number
  name: string
  selectorName: string
  categories: string[]
  value: PluginID
  icon?: string
  description?: string
  githubRepoUrl?: string
  isInstalled: boolean
  isPremium: boolean
  createdAt: string
  starters: ChatStarter[]
}

export interface Plugin {
  id: PluginID
}

export enum PluginID {
  NONE = "none",
  CODE_LLM = "codellm",
  AUTO_PLUGIN_SELECTOR = "autopluginselector",
  CVEMAP = "cvemap",
  GOLINKFINDER = "golinkfinder",
  NUCLEI = "nuclei",
  SUBFINDER = "subfinder",
  KATANA = "katana",
  HTTPX = "httpx",
  NAABU = "naabu",
  GAU = "gau",
  ALTERX = "alterx",
  DNSX = "dnsx",
  WEB_BROWSE = "webbrowse",
  ENHANCED_SEARCH = "enhancedsearch",
  WEB_SCRAPER = "webscraper",
  PLUGINS_STORE = "pluginselector"
}

export const pluginHelp = (plugin: PluginID): string => {
  return plugin === PluginID.WEB_SCRAPER
    ? `<USER HELP>If the user asks for help or webscraper help, use the following information: The web scrapper plugin is active. The Web Scrapper plugin works automatically. Any url added in the chat by the user will be automatically scraped and converted into markdown and added as a source and added to the context of the conversation.</USER HELP>`
    : ""
}

export const Plugins: Record<PluginID, Plugin> = Object.fromEntries(
  Object.values(PluginID).map(id => [id, { id }])
) as Record<PluginID, Plugin>

export const PluginList = Object.values(Plugins)

type PluginUrls = Record<string, string>

export const pluginUrls: PluginUrls = {
  HACKERGPT: "https://github.com/Hacker-GPT/HackerGPT-2.0",
  CVEMAP: "https://github.com/projectdiscovery/cvemap",
  SUBFINDER: "https://github.com/projectdiscovery/subfinder",
  GOLINKFINDER: "https://github.com/0xsha/GoLinkFinder",
  NUCLEI: "https://github.com/projectdiscovery/nuclei",
  KATANA: "https://github.com/projectdiscovery/katana",
  HTTPX: "https://github.com/projectdiscovery/httpx",
  NAABU: "https://github.com/projectdiscovery/naabu",
  GAU: "https://github.com/lc/gau",
  ALTERX: "https://github.com/projectdiscovery/alterx",
  DNSX: "https://github.com/projectdiscovery/dnsx"
}
