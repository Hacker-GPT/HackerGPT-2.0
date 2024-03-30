import { handleCvemapRequest } from "./cvemap/cvemap.content"
import { handleGoLinkFinderRequest } from "./golinkfinder/golinkfinder.content"

type ToolUrls = {
  [key: string]: string
}

export const toolUrls: ToolUrls = {
  Cvemap: "https://github.com/projectdiscovery/cvemap",
  Subfinder: "https://github.com/projectdiscovery/subfinder",
  GoLinkFinder: "https://github.com/0xsha/GoLinkFinder",
  Nuclei: "https://github.com/projectdiscovery/nuclei",
  Katana: "https://github.com/projectdiscovery/katana",
  HttpX: "https://github.com/projectdiscovery/httpx",
  Naabu: "https://github.com/projectdiscovery/naabu",
  Gau: "https://github.com/lc/gau",
  Alterx: "https://github.com/projectdiscovery/alterx"
}

type pluginHandlerFunction = (
  lastMessage: any,
  enableFeature: boolean,
  OpenAIStream: any,
  model: string,
  messagesToSend: any,
  answerMessage: any,
  invokedByToolId: boolean
) => Promise<any>

type pluginIdToHandlerMapping = {
  [key: string]: pluginHandlerFunction
}

export const pluginIdToHandlerMapping: pluginIdToHandlerMapping = {
  cvemap: handleCvemapRequest,
  golinkfinder: handleGoLinkFinderRequest
}
