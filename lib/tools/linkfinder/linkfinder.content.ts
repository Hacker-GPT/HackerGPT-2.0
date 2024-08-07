import { Message } from "@/types/chat"

import { createGKEHeaders } from "../../plugins/chatpluginhandlers"

import { displayHelpGuideForLinkFinder } from "@/lib/plugins/plugin-helper/help-guides"
import { transformUserQueryToLinkFinderCommand } from "@/lib/plugins/plugin-helper/transform-query-to-command"
import { handlePluginStreamError } from "@/lib/plugins/plugin-helper/plugin-stream"
import {
  createResponseString,
  processToolCommand
} from "@/lib/tools/tool-helper/tools-stream"

interface LinkFinderParams {
  domain: string[]
  output: string
  error: string | null
}

const parseLinkFinderCommandLine = (input: string): LinkFinderParams => {
  const MAX_INPUT_LENGTH = 1000
  const MAX_ARRAY_SIZE = 1

  const params: LinkFinderParams = {
    domain: [],
    output: "",
    error: null
  }

  if (input.length > MAX_INPUT_LENGTH) {
    params.error = `🚨 Input command is too long`
    return params
  }

  const trimmedInput = input.trim().toLowerCase()
  const args = trimmedInput.split(" ")
  args.shift()

  for (let i = 0; i < args.length; i++) {
    try {
      switch (args[i]) {
        case "--domain":
          params.domain = args[++i].split(",")
          if (params.domain.length > MAX_ARRAY_SIZE) {
            params.error = `🚨 Too many elements in domain array`
            return params
          }
          break
        case "--output":
          if (args[++i]) {
            params.output = args[i].trim()
          } else {
            params.error = `🚨 Output flag provided without value`
            return params
          }
          break
        default:
          params.error = `🚨 Invalid or unrecognized flag: ${args[i]}`
          break
      }
    } catch (error) {
      if (error instanceof Error) {
        return { ...params, error: error.message }
      }
    }
  }

  if (!params.domain || params.domain.length === 0) {
    params.error = `🚨 No target domain/URL provided`
    return params
  }

  return params
}

export async function handleLinkfinderRequest(
  lastMessage: Message,
  enableLinkFinderFeature: boolean,
  OpenRouterStream: any,
  messagesToSend: Message[],
  invokedByToolId: boolean
) {
  if (!enableLinkFinderFeature) {
    return new Response("The Link Finder is disabled.")
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
            transformUserQueryToLinkFinderCommand,
            OpenRouterStream,
            messagesToSend,
            sendMessage
          )
        } catch (error) {
          return new Response(`Error processing AI response: ${error}`)
        }
      }

      const parts = lastMessage.content.split(" ")
      if (
        parts.includes("-h") ||
        parts.includes("-help") ||
        parts.includes("--help")
      ) {
        sendMessage(displayHelpGuideForLinkFinder(), true)
        controller.close()
        return
      }

      const params = parseLinkFinderCommandLine(lastMessage.content)

      if (params.error) {
        handlePluginStreamError(
          params.error,
          invokedByToolId,
          sendMessage,
          controller
        )
        return
      }

      let linkfinderUrl = `${process.env.SECRET_GKE_PLUGINS_BASE_URL}/api/chat/plugins/golinkfinder?`

      if (Array.isArray(params.domain)) {
        const targetsString = params.domain.join(" ")
        linkfinderUrl += `domain=${encodeURIComponent(targetsString)}`
      }

      sendMessage(
        "🚀 URL extraction process initiated. This may take a minute to complete.",
        true
      )
      const startTime = Date.now()

      const intervalId = setInterval(() => {
        sendMessage("⏳ Still working on it, please hold on...", true)
      }, 15000)

      try {
        const linkfinderResponse = await fetch(linkfinderUrl, {
          method: "GET",
          headers: {
            Authorization: `${process.env.SECRET_AUTH_PLUGINS}`
          }
        })

        let linkfinderData = await linkfinderResponse.text()

        let urlsFormatted = processLinkFinderData(linkfinderData)

        if (!urlsFormatted || urlsFormatted.length === 0) {
          const noDataMessage = `🔍 Didn't find any URLs based on the provided command.`
          clearInterval(intervalId)
          sendMessage(noDataMessage, true)
          controller.close()
          return new Response(noDataMessage)
        }

        clearInterval(intervalId)
        sendMessage(
          "✅ URL extraction process is done! Now processing the results...",
          true
        )

        const endTime = Date.now()
        const responseString = createResponseString(
          "Link Finder",
          params.domain,
          urlsFormatted.trim(),
          startTime,
          endTime
        )

        sendMessage(responseString, true)

        controller.close()
      } catch (error) {
        clearInterval(intervalId)
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

const processLinkFinderData = (data: string): string => {
  const lines = data.split("\n")
  const processedLines = lines
    .filter(
      line =>
        !line.includes("Still processing...") &&
        !line.includes("goLinkFinder process completed.") &&
        !line.includes("Starting goLinkFinder process...")
    )
    .map(line => {
      if (line.startsWith("data: ")) {
        return line.replace("data: ", "").trim()
      }
      return line.trim()
    })
    .filter(line => line !== "")

  return processedLines.join("\n")
}
