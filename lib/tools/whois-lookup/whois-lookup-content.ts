import { Message } from "@/types/chat"
import {
  createGKEHeaders,
  getCommandFromAIResponse,
  processAIResponseAndUpdateMessage
} from "../../plugins/chatpluginhandlers"
import {
  WhoisLookupParams,
  WHOIS_MAX_INPUT_LENGTH,
  WHOIS_DEFAULT_PARAMS
} from "@/lib/tools/tools-helper/tools-flags"

import { displayHelpGuideForWhoisLookup } from "@/lib/plugins/plugin-helper/help-guides"
import { transformUserQueryToWhoisLookupCommand } from "@/lib/plugins/plugin-helper/transform-query-to-command"
import { handlePluginStreamError } from "@/lib/plugins/plugin-helper/plugin-stream"
import {
  createResponseString,
  makeToolRequest,
  processStreamResponse
} from "@/lib/tools/tools-helper/tools-stream"

const parseCommandLine = (input: string): WhoisLookupParams => {
  if (input.length > WHOIS_MAX_INPUT_LENGTH) {
    return { ...WHOIS_DEFAULT_PARAMS, error: `🚨 Input command is too long` }
  }

  const args = input.trim().split(/\s+/).slice(1)
  const params = { ...WHOIS_DEFAULT_PARAMS }

  for (let i = 0; i < args.length; i++) {
    const flag = args[i]

    if (flag !== "-t" && flag !== "--target") {
      return { ...params, error: `🚨 Invalid or unrecognized flag: ${flag}` }
    }

    const value = args[++i]
    if (value === undefined) {
      return { ...params, error: `🚨 Missing value for flag: ${flag}` }
    }

    try {
      // Simple validation for domain or IP address
      if (
        !/^([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}$/.test(
          value
        ) &&
        !/^(\d{1,3}\.){3}\d{1,3}$/.test(value)
      ) {
        throw new Error(
          `Invalid target: ${value}. Must be a valid domain name or IP address`
        )
      }
      params.target = value
    } catch (error: any) {
      return { ...params, error: `🚨 ${error.message}` }
    }
  }

  if (!params.target) {
    return { ...params, error: `🚨 Error: Target is required.` }
  }

  return params
}

export async function handleWhoisRequest(
  lastMessage: Message,
  enableWhoisLookup: boolean,
  OpenRouterStream: any,
  messagesToSend: Message[],
  invokedByToolId: boolean
) {
  if (!enableWhoisLookup) {
    return new Response("The Whois Lookup is disabled.")
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
            transformUserQueryToWhoisLookupCommand,
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
        sendMessage(displayHelpGuideForWhoisLookup(), true)
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

      let whoisLookupUrl = `${process.env.SECRET_GKE_TOOLS_BASE_URL}/api/chat/tools/whois-lookup`
      sendMessage("🚀 Starting the scan.", true)
      const startTime = Date.now()

      try {
        const whoisLookupResponse = await makeToolRequest(
          whoisLookupUrl,
          params,
          `${process.env.SECRET_AUTH_TOOLS}`
        )

        const { scanError } = await processStreamResponse(
          whoisLookupResponse,
          sendMessage
        )

        if (scanError) {
          sendMessage(`🚨 ${scanError}`, true)
          controller.close()
          return
        }

        const endTime = Date.now()
        const responseString = createResponseString(
          "Whois Lookup",
          params.target,
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
