import { Message } from "@/types/chat"
import {
  createGKEHeaders,
  getCommandFromAIResponse,
  processAIResponseAndUpdateMessage
} from "../../plugins/chatpluginhandlers"

import { displayHelpGuideForSSLScanner } from "@/lib/plugins/plugin-helper/help-guides"
import { transformUserQueryToSSLScannerCommand } from "@/lib/plugins/plugin-helper/transform-query-to-command"
import { handlePluginStreamError } from "@/lib/plugins/plugin-helper/plugin-stream"
import {
  createResponseString,
  makeToolRequest,
  processStreamResponse
} from "@/lib/tools/tools-helper/tools-stream"

interface SSLScannerParams {
  host: string[]
  scanType: "light" | "deep" | "custom"
  enableVulnerabilityChecks: boolean
  port: string
  topPorts: string
  error: string | null
}

const parseCommandLine = (input: string): SSLScannerParams => {
  const MAX_INPUT_LENGTH = 500
  const params: SSLScannerParams = {
    host: [],
    scanType: "light",
    enableVulnerabilityChecks: true,
    port: "",
    topPorts: "",
    error: null
  }

  if (input.length > MAX_INPUT_LENGTH) {
    return { ...params, error: `🚨 Input command is too long` }
  }

  const args = input.trim().toLowerCase().split(/\s+/).slice(1)
  const flagMap: { [key: string]: keyof SSLScannerParams } = {
    "-host": "host",
    "-st": "scanType",
    "-scan-type": "scanType",
    "-no-vuln-check": "enableVulnerabilityChecks",
    "-p": "port",
    "-port": "port",
    "-tp": "topPorts",
    "-top-ports": "topPorts"
  }

  for (let i = 0; i < args.length; i++) {
    const flag = args[i]
    const param = flagMap[flag]

    if (!param) {
      return { ...params, error: `🚨 Invalid or unrecognized flag: ${flag}` }
    }

    if (param === "host") {
      params.host = args[++i].split(",")
    } else if (param === "scanType") {
      const value = args[++i]
      if (!["light", "deep", "custom"].includes(value)) {
        return { ...params, error: `🚨 Invalid scan type: ${value}` }
      }
      params.scanType = value as "light" | "deep" | "custom"
    } else if (param === "enableVulnerabilityChecks") {
      params[param] = false
    } else if (param === "port" || param === "topPorts") {
      const value = args[++i]
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
      params[param] = value
    }
  }

  if (!params.host.length) {
    return { ...params, error: `🚨 Error: -host parameter is required.` }
  }

  return params
}

export async function handleSslscannerRequest(
  lastMessage: Message,
  enableSSLScanner: boolean,
  OpenRouterStream: any,
  messagesToSend: Message[],
  invokedByToolId: boolean
) {
  if (!enableSSLScanner) {
    return new Response("The SSL/TLS Scanner is disabled.")
  }

  let aiResponse = ""

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
          for await (const chunk of processAIResponseAndUpdateMessage(
            lastMessage,
            transformUserQueryToSSLScannerCommand,
            OpenRouterStream,
            messagesToSend
          )) {
            sendMessage(chunk, false)
            aiResponse += chunk
          }

          sendMessage("\n\n")
          lastMessage.content = getCommandFromAIResponse(
            lastMessage,
            aiResponse
          )
        } catch (error) {
          return new Response(`Error processing AI response: ${error}`)
        }
      }

      const parts = lastMessage.content.split(" ")
      if (parts.includes("-h") || parts.includes("-help")) {
        sendMessage(displayHelpGuideForSSLScanner(), true)
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

      let sslScannerUrl = `${process.env.SECRET_GKE_TOOLS_BASE_URL}/api/chat/tools/ssl-scanner`
      sendMessage("🚀 Starting the scan. It might take a minute.", true)
      const startTime = Date.now()

      try {
        const sslScannerResponse = await makeToolRequest(
          sslScannerUrl,
          params,
          `${process.env.SECRET_AUTH_TOOLS}`
        )

        const { scanError } = await processStreamResponse(
          sslScannerResponse,
          sendMessage
        )

        if (scanError) {
          sendMessage(`🚨 ${scanError}`, true)
          controller.close()
          return
        }

        const endTime = Date.now()
        const responseString = createResponseString(
          "SSL/TLS Scanner",
          "",
          "",
          startTime,
          endTime,
          false
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
