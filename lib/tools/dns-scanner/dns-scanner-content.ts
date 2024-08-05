import { Message } from "@/types/chat"
import { createGKEHeaders } from "../../plugins/chatpluginhandlers"
import {
  DNSScannerParams,
  DNS_SCANNER_MAX_INPUT_LENGTH,
  DNS_SCANNER_DEFAULT_PARAMS
} from "@/lib/tools/tool-helper/tools-flags"

import { displayHelpGuideForDNSScanner } from "@/lib/tools/tool-helper/tool-help-guides"
import { transformUserQueryToDNSScannerCommand } from "@/lib/tools/tool-helper/transform-query-to-command"
import { handlePluginStreamError } from "@/lib/plugins/plugin-helper/plugin-stream"
import {
  createResponseString,
  makeToolRequest,
  processStreamResponse,
  processToolCommand
} from "@/lib/tools/tool-helper/tools-stream"

const parseCommandLine = (input: string): DNSScannerParams => {
  if (input.length > DNS_SCANNER_MAX_INPUT_LENGTH) {
    return {
      ...DNS_SCANNER_DEFAULT_PARAMS,
      error: `🚨 Input command is too long`
    }
  }

  const args = input.trim().split(/\s+/).slice(1)
  const params = { ...DNS_SCANNER_DEFAULT_PARAMS }

  for (let i = 0; i < args.length; i++) {
    const flag = args[i]

    switch (flag) {
      case "-t":
      case "-target":
        const value = args[++i]
        if (value === undefined) {
          return { ...params, error: `🚨 Missing value for flag: ${flag}` }
        }
        params.target = value
        break
      case "-z":
      case "-zone-transfer":
        params.zoneTransfer = args[++i]?.toLowerCase() !== "false"
        break
      default:
        return { ...params, error: `🚨 Invalid or unrecognized flag: ${flag}` }
    }
  }

  if (!params.target) {
    return { ...params, error: `🚨 Error: Target is required.` }
  }

  return params
}

export async function handleDnsscannerRequest(
  lastMessage: Message,
  enableDNSScanner: boolean,
  OpenRouterStream: any,
  messagesToSend: Message[],
  invokedByToolId: boolean
) {
  if (!enableDNSScanner) {
    return new Response("The DNS Scanner is disabled.")
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
            transformUserQueryToDNSScannerCommand,
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
        sendMessage(displayHelpGuideForDNSScanner(), true)
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

      let dnsScannerUrl = `${process.env.SECRET_GKE_TOOLS_BASE_URL}/api/chat/tools/dns-scanner`
      sendMessage("🚀 Starting the scan.", true)
      const startTime = Date.now()

      try {
        const dnsScannerResponse = await makeToolRequest(
          dnsScannerUrl,
          params,
          `${process.env.SECRET_AUTH_TOOLS}`
        )

        const { scanError, fileContent } = await processStreamResponse(
          dnsScannerResponse,
          sendMessage
        )

        if (scanError) {
          sendMessage(`🚨 ${scanError}`, true)
          controller.close()
          return
        }

        const processedContent = processFileContent(fileContent)

        if (!processedContent || processedContent.trim() === "") {
          sendMessage("🔍 No results were found during the scan.", true)
          controller.close()
          return
        }

        sendMessage("✅ Scan done! Now processing the results...", true)

        const endTime = Date.now()
        const responseString = createResponseString(
          "DNS Scanner",
          params.target,
          processedContent,
          startTime,
          endTime,
          true,
          true,
          true
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

function processFileContent(fileContent: string | any[]): string {
  try {
    const contentArray =
      typeof fileContent === "string" ? JSON.parse(fileContent) : fileContent
    if (Array.isArray(contentArray)) {
      if (contentArray.length > 0 && "arguments" in contentArray[0]) {
        contentArray.shift() // Remove the first element containing 'arguments'
      }

      // Group the array items by their 'type'
      const groupedContent = contentArray.reduce(
        (acc, item) => {
          if (!acc[item.type]) {
            acc[item.type] = []
          }
          acc[item.type].push(item)
          return acc
        },
        {} as Record<string, any[]>
      )

      // Create a table for each group
      const tables = Object.entries(groupedContent).map(([type, items]) => {
        return `### ${type.charAt(0).toUpperCase() + type.slice(1)} Results\n\n${createTable(items as any[])}`
      })

      return tables.join("\n\n")
    }
    return typeof fileContent === "string"
      ? fileContent
      : JSON.stringify(fileContent, null, 2)
  } catch (error) {
    console.error("Error processing fileContent:", error)
    return typeof fileContent === "string"
      ? fileContent
      : JSON.stringify(fileContent)
  }
}

function createTable(data: any[]): string {
  if (data.length === 0) return ""

  const headers = Object.keys(data[0]).filter(key => key !== "type")
  const rows = data.map(item => headers.map(header => item[header] || ""))

  const columnWidths = headers.map((header, index) =>
    Math.max(header.length, ...rows.map(row => String(row[index]).length))
  )

  const createRow = (values: string[]) =>
    `| ${values.map((v, i) => v.padEnd(columnWidths[i])).join(" | ")} |`

  const headerRow = createRow(headers)
  const separatorRow = `|${columnWidths.map(w => "-".repeat(w + 2)).join("|")}|`
  const dataRows = rows.map(row => createRow(row.map(String)))

  return [headerRow, separatorRow, ...dataRows].join("\n")
}
