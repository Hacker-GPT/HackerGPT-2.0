import { handleCvemapRequest } from "./cvemap/cvemap.content"

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
  cvemap: handleCvemapRequest
}
