import { Message } from "@/types/chat"
import {
  createGKEHeaders
} from "../../plugins/chatpluginhandlers"

import { displayHelpGuideForPortScanner } from "@/lib/plugins/plugin-helper/help-guides"
import { transformUserQueryToPortScannerCommand } from "@/lib/plugins/plugin-helper/transform-query-to-command"
import { handlePluginStreamError } from "@/lib/plugins/plugin-helper/plugin-stream"
import {
  makeToolRequest,
  processStreamResponse,
  processToolCommand
} from "@/lib/tools/tool-helper/tools-stream"
import { createResponseString } from "@/lib/tools/tool-helper/tools-stream"

interface PortScannerParams {
  host: string[]
  scanType: "light" | "deep" | "custom"
  port: string
  topPorts: string
  noSvc: boolean
  error: string | null
}

const parseCommandLine = (input: string): PortScannerParams => {
  const MAX_INPUT_LENGTH = 500
  const params: PortScannerParams = {
    host: [],
    scanType: "light",
    port: "",
    topPorts: "",
    noSvc: false,
    error: null
  }

  if (input.length > MAX_INPUT_LENGTH) {
    return { ...params, error: `🚨 Input command is too long` }
  }

  const args = input.trim().toLowerCase().split(/\s+/).slice(1)
  const flagMap: { [key: string]: keyof PortScannerParams } = {
    "-host": "host",
    "-st": "scanType",
    "-scan-type": "scanType",
    "-p": "port",
    "-port": "port",
    "-tp": "topPorts",
    "-top-ports": "topPorts",
    "-no-svc": "noSvc"
  }

  for (let i = 0; i < args.length; i++) {
    const flag = args[i]
    const param = flagMap[flag]

    if (!param) {
      return { ...params, error: `🚨 Invalid or unrecognized flag: ${flag}` }
    }

    const value = args[++i]

    if (param === "host") {
      params.host = value.split(",")
    } else if (param === "scanType") {
      if (!["light", "deep", "custom"].includes(value)) {
        return { ...params, error: `🚨 Invalid scan type: ${value}` }
      }
      params.scanType = value as "light" | "deep" | "custom"
    } else if (param === "port" || param === "topPorts" || param === "noSvc") {
      if (params.scanType !== "custom") {
        return {
          ...params,
          error: `🚨 The option "${flag}" is only allowed for custom scan type.`
        }
      }
      if (param === "topPorts" && !["full", "100", "1000"].includes(value)) {
        return {
          ...params,
          error: `🚨 Invalid value for -top-ports: ${value}. Allowed values are "full", "100", "1000".`
        }
      }
      if (param === "noSvc") {
        params.noSvc = true
      } else {
        params[param] = value
      }
    }
  }

  if (!params.host.length) {
    return { ...params, error: `🚨 Error: -host parameter is required.` }
  }

  return params
}
export async function handlePortscannerRequest(
  lastMessage: Message,
  enablePortScanner: boolean,
  OpenRouterStream: any,
  messagesToSend: Message[],
  invokedByToolId: boolean
) {
  if (!enablePortScanner) {
    return new Response("The Port Scanner is disabled.")
  }

  const headers = createGKEHeaders()

  const stream = new ReadableStream({
    async start(controller) {
      const sendMessage = (
        data: string,
        addExtraLineBreaks: boolean = false
      ) => {
        const formattedData = addExtraLineBreaks ? `${data}\n\n` : data
        controller.enqueue(new TextEncoder().encode(formattedData))
      }

      if (invokedByToolId) {
        try {
          await processToolCommand(
            lastMessage,
            transformUserQueryToPortScannerCommand,
            OpenRouterStream,
            messagesToSend,
            sendMessage
          )
        } catch (error) {
          return new Response(`Error processing AI response: ${error}`)
        }
      }

      const parts = lastMessage.content.split(" ")
      if (parts.includes("-h") || parts.includes("-help")) {
        sendMessage(displayHelpGuideForPortScanner(), true)
        controller.close()
        return
      }

      const params = parseCommandLine(lastMessage.content)

      if (params.error) {
        handlePluginStreamError(
          params.error,
          invokedByToolId,
          sendMessage,
          controller
        )
        return
      }

      let portScannerUrl = `${process.env.SECRET_GKE_TOOLS_BASE_URL}/api/chat/tools/port-scanner`
      sendMessage("🚀 Starting the scan. It might take a minute.", true)
      const startTime = Date.now()

      try {
        const portScannerResponse = await makeToolRequest(
          portScannerUrl,
          params,
          `${process.env.SECRET_AUTH_TOOLS}`
        )

        const { fileContent, scanError } = await processStreamResponse(
          portScannerResponse,
          sendMessage
        )

        if (scanError) {
          sendMessage(`🚨 ${scanError}`, true)
          controller.close()
          return
        }

        if (!fileContent || fileContent.trim() === "") {
          sendMessage("🔍 No open ports were found during the scan.", true)
          controller.close()
          return
        }

        sendMessage("✅ Scan done! Now processing the results...", true)

        const endTime = Date.now()
        const responseString = createResponseString(
          "Port Scanner",
          params.host,
          fileContent.trim(),
          startTime,
          endTime
        )
        sendMessage(responseString, true)

        controller.close()
      } catch (error) {
        let errorMessage =
          "🚨 There was a problem during the scan. Please try again."
        if (error instanceof Error) {
          errorMessage = `🚨 Error: ${error.message}`
        }
        sendMessage(errorMessage, true)
        controller.close()
        return new Response(errorMessage)
      }
    }
  })

  return new Response(stream, { headers })
}
