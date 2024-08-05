import { Message } from "@/types/chat"
import {
  createGKEHeaders
} from "../../plugins/chatpluginhandlers"
import {
  WafDetectorParams,
  WAF_DETECTOR_MAX_INPUT_LENGTH,
  WAF_DETECTOR_DEFAULT_PARAMS
} from "@/lib/tools/tool-helper/tools-flags"

import { displayHelpGuideForWAFDetector } from "@/lib/tools/tool-helper/tool-help-guides"
import { transformUserQueryToWAFDetectorCommand } from "@/lib/tools/tool-helper/transform-query-to-command"
import { handlePluginStreamError } from "@/lib/plugins/plugin-helper/plugin-stream"
import {
  createResponseString,
  makeToolRequest,
  processStreamResponse,
  processToolCommand
} from "@/lib/tools/tool-helper/tools-stream"

const parseCommandLine = (input: string): WafDetectorParams => {
  if (input.length > WAF_DETECTOR_MAX_INPUT_LENGTH) {
    return {
      ...WAF_DETECTOR_DEFAULT_PARAMS,
      error: `🚨 Input command is too long`
    }
  }

  const args = input.trim().split(/\s+/).slice(1)
  const params = { ...WAF_DETECTOR_DEFAULT_PARAMS }

  for (let i = 0; i < args.length; i++) {
    const flag = args[i]

    if (flag !== "-t" && flag !== "--target") {
      return { ...params, error: `🚨 Invalid or unrecognized flag: ${flag}` }
    }

    const value = args[++i]
    if (value === undefined) {
      return { ...params, error: `🚨 Missing value for flag: ${flag}` }
    }

    params.target = value
  }

  if (!params.target) {
    return { ...params, error: `🚨 Error: Target is required.` }
  }

  return params
}

export async function handleWafdetectorRequest(
  lastMessage: Message,
  enableWAFDetector: boolean,
  OpenRouterStream: any,
  messagesToSend: Message[],
  invokedByToolId: boolean
) {
  if (!enableWAFDetector) {
    return new Response("The WAF Detector is disabled.")
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
            transformUserQueryToWAFDetectorCommand,
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
        sendMessage(displayHelpGuideForWAFDetector(), true)
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

      let wafDetectorUrl = `${process.env.SECRET_GKE_TOOLS_BASE_URL}/api/chat/tools/waf-detector`
      sendMessage("🚀 Starting the scan.", true)
      const startTime = Date.now()

      try {
        const wafDetectorResponse = await makeToolRequest(
          wafDetectorUrl,
          params,
          `${process.env.SECRET_AUTH_TOOLS}`
        )

        const { scanError, fileContent } = await processStreamResponse(
          wafDetectorResponse,
          sendMessage
        )

        if (scanError) {
          sendMessage(`🚨 ${scanError}`, true)
          controller.close()
          return
        }

        if (!fileContent || fileContent.trim() === "") {
          sendMessage("🔍 No results were found during the scan.", true)
          controller.close()
          return
        }

        sendMessage("✅ Scan done! Now processing the results...", true)

        const endTime = Date.now()
        const responseString = createResponseString(
          "WAF Detector",
          params.target,
          fileContent,
          startTime,
          endTime,
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
