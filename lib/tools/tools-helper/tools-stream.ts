import endent from "endent"

interface StreamProcessResult {
  fileContent: string
  scanError: string | null
}

export async function processStreamResponse(
  response: Response,
  sendMessage: (data: string, addExtraLineBreaks?: boolean) => void
): Promise<StreamProcessResult> {
  const reader = response.body?.getReader()
  const decoder = new TextDecoder()
  let fileContent = ""
  let scanError: string | null = null
  let isFileContent = false
  let hasOutputContent = false
  let buffer = ""
  let isFirstOutput = true

  try {
    while (true) {
      const { done, value } = (await reader?.read()) ?? {}
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      buffer += chunk
      const lines = buffer.split("\n")
      buffer = lines.pop() || ""

      for (const line of lines) {
        if (line.startsWith("SCAN_ERROR:")) {
          scanError = line.substring(11).trim()
          return { fileContent: "", scanError }
        }

        if (line.includes("--- FILE CONTENT START ---")) {
          isFileContent = true
          continue
        }

        if (isFileContent) {
          fileContent += line + "\n"
        } else {
          if (isFirstOutput) {
            sendMessage("**Raw output:**\n```terminal\n", false)
            isFirstOutput = false
            hasOutputContent = true
          }
          sendMessage(line + "\n", false)
        }
      }
    }

    // Process any remaining content in the buffer
    if (buffer) {
      if (isFileContent) {
        fileContent += buffer
      } else {
        if (isFirstOutput) {
          sendMessage("**Raw output:**\n```terminal\n", false)
          hasOutputContent = true
        }
        sendMessage(buffer, false)
      }
    }
  } finally {
    if (hasOutputContent) {
      sendMessage("```\n", false)
    }
  }

  return { fileContent, scanError }
}

export function createRequestBody<T extends Record<string, any>>(
  params: T
): Partial<T> {
  return Object.fromEntries(
    Object.entries(params).filter(
      ([_, value]) =>
        (Array.isArray(value) && value.length > 0) ||
        (typeof value === "boolean" && value) ||
        (typeof value === "number" && value > 0) ||
        (typeof value === "string" && value.length > 0)
    )
  ) as Partial<T>
}

export async function makeToolRequest<T extends Record<string, any>>(
  url: string,
  params: T,
  authToken: string
): Promise<Response> {
  const requestBody = createRequestBody(params)
  return fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: authToken
    },
    body: JSON.stringify(requestBody)
  })
}

export function createResponseString(
  toolName: string,
  target: string | string[],
  results: string,
  startTime: number,
  endTime: number
): string {
  const calculateScanDuration = (
    startTime: number,
    endTime: number
  ): number => {
    return Math.round((endTime - startTime) / 1000)
  }

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds} sec`
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes} min ${remainingSeconds} sec`
  }

  const scanDuration = calculateScanDuration(startTime, endTime)
  const formatDate = (date: Date) => {
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false
    })
  }

  const startTimeString = formatDate(new Date(startTime))
  const endTimeString = formatDate(new Date(endTime))

  return endent`
    # ${toolName} Results

    **Target**: ${Array.isArray(target) ? target.join(", ") : target}

    ## Results:
    \`\`\`
    ${results.trim()}
    \`\`\`

    ### Scan Information:
    - Start time:   ${startTimeString}
    - Finish time:  ${endTimeString}
    - Scan duration: ${formatDuration(scanDuration)}
  `
}
