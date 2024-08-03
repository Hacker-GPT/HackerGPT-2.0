import { Message } from "@/types/chat"
import {
  createGKEHeaders,
  getCommandFromAIResponse,
  processAIResponseAndUpdateMessage
} from "../chatpluginhandlers"

import { displayHelpGuideForSubdomainFinder } from "@/lib/plugins/plugin-helper/help-guides"
import { transformUserQueryToSubdomainFinderCommand } from "@/lib/plugins/plugin-helper/transform-query-to-command"
import { handlePluginStreamError } from "@/lib/plugins/plugin-helper/plugin-stream"

interface SubfinderParams {
  domain: string[]
  onlyActive: boolean
  json: boolean
  ip: boolean
  // FILE OUTPUT
  output: string
  error: string | null
}

const parseCommandLine = (input: string) => {
  const MAX_INPUT_LENGTH = 500
  const maxDomainLength = 255

  const params: SubfinderParams = {
    domain: [],
    onlyActive: false,
    json: false,
    ip: false,
    // FILE OUTPUT
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

  const isValidDomain = (domain: string) =>
    /^[a-zA-Z0-9.-]+$/.test(domain) && domain.length <= maxDomainLength

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "-d":
      case "-domain":
        const domainArgs = args[++i].split(",")
        for (const domain of domainArgs) {
          if (
            isValidDomain(domain.trim()) &&
            domain.length <= maxDomainLength
          ) {
            params.domain.push(domain.trim())
          } else {
            params.error = `🚨 Invalid or too long domain provided (max ${maxDomainLength} characters)`
            return params
          }
        }
        break
      case "-nw":
      case "-active":
        params.onlyActive = true
        break
      case "-oj":
      case "-json":
        params.json = true
        break
      case "-oi":
      case "-ip":
        params.ip = true
        break
      case "-output":
        if (i + 1 < args.length) {
          params.output = args[++i]
        } else {
          params.error = `🚨 Output flag provided without value`
          return params
        }
        break
      default:
        params.error = `🚨 Invalid or unrecognized flag: ${args[i]}`
        return params
    }
  }

  if (!params.domain.length || params.domain.length === 0) {
    params.error = "🚨 Error: -d/-domain parameter is required."
  }

  return params
}

export async function handleSubfinderRequest(
  lastMessage: Message,
  enableSubfinderFeature: boolean,
  OpenRouterStream: any,
  messagesToSend: Message[],
  invokedByToolId: boolean
) {
  if (!enableSubfinderFeature) {
    return new Response("The Subfinder is disabled.")
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
            transformUserQueryToSubdomainFinderCommand,
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
        sendMessage(displayHelpGuideForSubdomainFinder(), true)
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

      const subfinderBaseUrl = `${process.env.SECRET_GKE_PLUGINS_BASE_URL}/api/chat/plugins/subfinder`
      const queryParams = Object.entries(params)
        .filter(([key, value]) => {
          if (key === "domain") return value.length > 0
          if (key === "onlyActive" || key === "json" || key === "ip")
            return value === true
          return false
        })
        .map(([key, value]) => {
          if (key === "domain")
            return value
              .map((d: string) => `domain=${encodeURIComponent(d)}`)
              .join("&")
          return `${key}=true`
        })
        .join("&")

      const subfinderUrl = `${subfinderBaseUrl}?${queryParams}`

      sendMessage("🚀 Starting the scan. It might take a minute.", true)

      const intervalId = setInterval(() => {
        sendMessage("⏳ Still working on it, please hold on...", true)
      }, 15000)

      try {
        const subfinderResponse = await fetch(subfinderUrl, {
          method: "GET",
          headers: {
            Authorization: `${process.env.SECRET_AUTH_PLUGINS}`
          }
        })

        let subfinderData = await subfinderResponse.text()
        const processedData = processSubfinderData(subfinderData)

        if (!processedData.subdomains || processedData.count === 0) {
          const noDataMessage = `🔍 Alright, I've looked into "${params.domain.join(
            ", "
          )}" based on your command: "${
            lastMessage.content
          }". Turns out, there are no subdomains to report back on this time.`
          clearInterval(intervalId)
          sendMessage(noDataMessage, true)
          controller.close()
          return new Response(noDataMessage)
        }

        clearInterval(intervalId)
        sendMessage("✅ Scan done! Now processing the results...", true)

        const responseString = createResponseString(
          params.domain,
          processedData
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

const processSubfinderData = (
  data: string
): { subdomains: string; count: number } => {
  const lines = data
    .split("\n")
    .filter(line => !line.startsWith("data:") && line.trim() !== "")

  return {
    subdomains: lines.join("\n"),
    count: lines.length
  }
}

const createResponseString = (
  domain: string | string[],
  subfinderData: { subdomains: string; count: number }
) => {
  return (
    `# Subdomain Finder Results\n` +
    `**Target**: "${Array.isArray(domain) ? domain.join(", ") : domain}"\n\n` +
    `**Found ${subfinderData.count} subdomains**\n` +
    `## Results:\n` +
    "```\n" +
    subfinderData.subdomains +
    "\n" +
    "```\n"
  )
}
