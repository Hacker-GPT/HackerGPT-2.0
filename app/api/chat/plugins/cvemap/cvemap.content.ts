import { Message } from "@/types/chat"
import { pluginUrls } from "@/types/plugins"
import endent from "endent"

import {
  createGKEHeaders,
  getCommandFromAIResponse,
  processAIResponseAndUpdateMessage,
  truncateData
} from "../chatpluginhandlers"

import { displayHelpGuideForCvemap } from "../plugin-helper/help-guides"
import { transformUserQueryToCvemapCommand } from "../plugin-helper/transform-query-to-command"
import { handlePluginStreamError } from "../plugin-helper/plugin-stream"

interface CvemapParams {
  ids?: string[]
  cwes?: string[]
  vendors?: string
  products?: string
  excludeProducts?: string
  severity?: string
  cvssScores?: string
  cpe?: string
  epssScores?: string
  epssPercentiles?: string
  age?: string
  assignees?: string
  vulnerabilityStatus?: string
  search?: string
  kev?: boolean
  template?: boolean
  poc?: boolean
  hackerone?: boolean
  remote?: boolean
  fieldsToDisplay?: string
  excludeFields?: string
  listIdsOnly?: boolean
  limit?: number
  offset?: number
  json?: boolean
  error?: string | null
}

const parseCommandLine = (input: string): CvemapParams => {
  const MAX_INPUT_LENGTH = 500

  const params: CvemapParams = {
    ids: [],
    cwes: [],
    limit: 25,
    offset: 0,
    json: false
  }

  if (input.length > MAX_INPUT_LENGTH) {
    params.error = "🚨 Input command is too long."
    return params
  }

  const trimmedInput = input.trim()
  const argsRegex = /'[^']*'|[^\s]+/g
  const args =
    trimmedInput.match(argsRegex)?.map(arg => arg.replace(/^'|'$/g, "")) || []
  args.shift()

  const flagMap: Record<string, keyof CvemapParams> = {
    "-id": "ids",
    "-cwe": "cwes",
    "-cwe-id": "cwes",
    "-v": "vendors",
    "-vendor": "vendors",
    "-p": "products",
    "-product": "products",
    "-eproduct": "excludeProducts",
    "-s": "severity",
    "-severity": "severity",
    "-cs": "cvssScores",
    "-cvss-score": "cvssScores",
    "-c": "cpe",
    "-cpe": "cpe",
    "-es": "epssScores",
    "-epss-score": "epssScores",
    "-ep": "epssPercentiles",
    "-epss-percentile": "epssPercentiles",
    "-age": "age",
    "-a": "assignees",
    "-assignee": "assignees",
    "-vs": "vulnerabilityStatus",
    "-vstatus": "vulnerabilityStatus",
    "-q": "search",
    "-search": "search",
    "-f": "fieldsToDisplay",
    "-fields": "fieldsToDisplay",
    "-fe": "excludeFields",
    "-exclude": "excludeFields",
    "-l": "limit",
    "-limit": "limit",
    "-offset": "offset"
  }

  const booleanFlags: Record<string, keyof CvemapParams> = {
    "-k": "kev",
    "-kev": "kev",
    "-t": "template",
    "-template": "template",
    "-poc": "poc",
    "-h1": "hackerone",
    "-hackerone": "hackerone",
    "-re": "remote",
    "-remote": "remote",
    "-lsi": "listIdsOnly",
    "-list-id": "listIdsOnly",
    "-j": "json",
    "-json": "json"
  }

  const encounteredFlags: Set<string> = new Set()
  const repeatableFlags: Set<string> = new Set(["-id", "-cwe", "-cwe-id"])
  const allRecognizedFlags = new Set([
    ...Object.keys(flagMap),
    ...Object.keys(booleanFlags)
  ])

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]

    if (!allRecognizedFlags.has(arg)) {
      params.error = `🚨 Unrecognized flag: ${arg}`
      return params
    }

    if (flagMap[arg]) {
      if (encounteredFlags.has(arg) && !repeatableFlags.has(arg)) {
        params.error = `🚨 Duplicate flag: ${arg}`
        return params
      }
      encounteredFlags.add(arg)

      const key = flagMap[arg]
      const value = args[++i]
      if (Array.isArray(params[key])) {
        ;(params[key] as string[]).push(...value.split(","))
      } else if (key === "limit") {
        const limit = parseInt(value, 10)
        if (!isNaN(limit)) {
          params.limit = Math.min(limit, 25)
        }
      } else if (key === "offset") {
        const offset = parseInt(value, 10)
        if (!isNaN(offset)) params.offset = offset
      } else {
        ;(params[key] as string) = value
      }
    } else if (booleanFlags[arg]) {
      if (encounteredFlags.has(arg)) {
        params.error = `🚨 Duplicate flag: ${arg}`
        return params
      }
      encounteredFlags.add(arg)

      params[booleanFlags[arg] as keyof CvemapParams] = true as any
    }
  }

  return params
}

export async function handleCvemapRequest(
  lastMessage: Message,
  enableCvemapFeature: boolean,
  OpenAIStream: any,
  model: string,
  messagesToSend: Message[],
  answerMessage: Message,
  invokedByToolId: boolean
) {
  if (!enableCvemapFeature) {
    return new Response("The CVEMap is disabled.")
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
            transformUserQueryToCvemapCommand,
            OpenAIStream,
            model,
            messagesToSend,
            answerMessage
          )) {
            sendMessage(chunk, false)
            aiResponse += chunk
          }

          sendMessage("\n\n")
          lastMessage.content = getCommandFromAIResponse(
            lastMessage,
            messagesToSend,
            aiResponse
          )
        } catch (error) {
          console.error(
            "Error processing AI response and updating message:",
            error
          )
          return new Response(`Error processing AI response: ${error}`)
        }
      }

      const parts = lastMessage.content.split(" ")
      if (parts.includes("-h") || parts.includes("-help")) {
        sendMessage(displayHelpGuideForCvemap(), true)
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

      let cvemapUrl = `${process.env.SECRET_GKE_PLUGINS_BASE_URL}/api/chat/plugins/cvemap`

      let requestBody: Partial<CvemapParams> = {}

      for (const [key, value] of Object.entries(params)) {
        if (
          (Array.isArray(value) && value.length > 0) ||
          (typeof value === "boolean" && value) ||
          (typeof value === "number" && value > 0) ||
          (typeof value === "string" && value.length > 0)
        ) {
          ;(requestBody as any)[key] = value
        }
      }

      const intervalId = setInterval(() => {
        sendMessage(
          "⏳ Searching in progress. We appreciate your patience.",
          true
        )
      }, 15000)

      try {
        const cvemapResponse = await fetch(cvemapUrl, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `${process.env.SECRET_AUTH_PLUGINS}`
          },
          body: JSON.stringify(requestBody)
        })

        let cvemapData = await cvemapResponse.text()
        cvemapData = processCvemapData(cvemapData)
        cvemapData = truncateData(cvemapData, 300000)

        if (!cvemapData || cvemapData.length <= 300) {
          sendMessage(
            "🔍 The search is complete. No CVE entries were found based on your parameters.",
            true
          )
          clearInterval(intervalId)
          controller.close()
          return new Response("No CVE entries found.")
        }

        clearInterval(intervalId)

        if (params.json && !cvemapData.includes("╭──────")) {
          const responseString = createResponseString(cvemapData)
          sendMessage(responseString, true)
          controller.close()
          return new Response(cvemapData)
        }

        const responseString = formatCvemapOutput(cvemapData)
        sendMessage(responseString, true)

        controller.close()
      } catch (error) {
        clearInterval(intervalId)
        let errorMessage =
          "🚨 An unexpected error occurred during the CVE scan. Please try again later."
        if (error instanceof Error) {
          errorMessage = `🚨 Error: ${error.message}. Please check your request or try again later.`
        }
        sendMessage(errorMessage, true)
        controller.close()
        return new Response(errorMessage)
      }
    }
  })

  return new Response(stream, { headers })
}

function formatCvemapOutput(output: string): string {
  const asciiArt = `
    ______   _____  ____ ___  ____  ____
   / ___/ | / / _ \\/ __ \\__ \\/ __ \\/ __ \\
  / /__ | |/ /  __/ / / / / / /_/ / /_/ /
  \\___/ |___/\\___/_/ /_/ /_/\\__,_/ .___/ 
                                /_/            
    projectdiscovery.io
  `

  const parsedOutput = JSON.parse(output).output

  return (
    `## CVE Details Report\n\n` +
    "```\n" +
    asciiArt +
    "\n" +
    parsedOutput +
    "\n```"
  )
}

const processCvemapData = (data: string) => {
  return data
    .split("\n")
    .filter(line => line && !line.startsWith("data:") && line.trim() !== "")
    .join("")
}

const createResponseString = (cvemapData: string) => {
  const outerData = JSON.parse(cvemapData)
  const data = JSON.parse(outerData.output)
  let markdownOutput = `# CVE Discovery\n\n`

  const formatTime = (timeValue: string | Date) => {
    return new Date(timeValue).toLocaleString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric"
    })
  }

  const addTimeField = (label: string, timeValue: string | Date) => {
    markdownOutput += `- **${label}**: ${formatTime(timeValue)}\n`
  }

  const addOptionalField = (label: string, value: string) => {
    if (value) markdownOutput += `- **${label}**: ${value}\n`
  }

  data.forEach(
    (cve: {
      nuclei_templates?: {
        created_at: string
        template_issue: string
        template_issue_type: string
        template_path: string
        template_pr: string
        template_url: string
        updated_at: string
      }
      cve_id: any
      cve_description: any
      severity: any
      cvss_score: any
      cvss_metrics: any
      epss_percentile: any
      published_at: any
      updated_at: any
      weaknesses: any
      cpe: any
      reference: any
      poc: any
      age_in_days: any
      vuln_status: any
      is_poc: any
      is_remote: any
      is_oss: any
      vulnerable_cpe: any
      vendor_advisory: any
      patch_url: any
      is_template: any
      is_exploited: any
      hackerone: any
      shodan: any
      oss: any
    }) => {
      const {
        nuclei_templates,
        cve_id,
        cve_description,
        severity,
        cvss_score,
        cvss_metrics,
        published_at,
        epss_percentile,
        updated_at,
        weaknesses,
        cpe,
        reference,
        poc,
        age_in_days,
        vuln_status,
        is_poc,
        is_remote,
        is_oss,
        vulnerable_cpe,
        vendor_advisory,
        patch_url,
        is_template,
        is_exploited,
        hackerone,
        shodan,
        oss
      } = cve

      markdownOutput += `## ${cve_id}\n`
      markdownOutput += `### Overview\n`
      markdownOutput += `- **Severity**: ${severity[0].toUpperCase() + severity.slice(1)} | **CVSS Score**: ${cvss_score}\n`
      markdownOutput += `- **CVSS Vector**: (${cvss_metrics?.cvss31?.vector})\n`
      if (weaknesses?.length) {
        markdownOutput += `- **Weaknesses**:\n`
        weaknesses.forEach(
          (w: { cwe_name: any; cwe_id: any }) =>
            (markdownOutput += `  - ${w.cwe_name || w.cwe_id}\n`)
        )
      }
      if (epss_percentile?.length) {
        markdownOutput += `- **EPSS**: ${epss_percentile} %\n`
      }
      let timeInfo = ""
      addOptionalField("**Days Since Publish**:", age_in_days)
      if (cve.published_at?.length)
        timeInfo += `**Published At:** ${formatTime(cve.published_at)}`
      if (cve.updated_at?.length)
        timeInfo += ` | **Updated At**: ${formatTime(cve.updated_at)}`
      if (timeInfo) markdownOutput += `- ${timeInfo}\n`
      if (cpe?.vendor || cpe?.product) {
        markdownOutput += `- **CPE**: ${cpe.vendor || "Unknown vendor"}:${cpe.product || "Unknown product"}\n`
      }

      markdownOutput += `### Description\n${cve_description}\n`

      if (reference?.length) {
        markdownOutput += `### References:\n`
        reference.forEach(
          (ref: any) => (markdownOutput += `  - [${ref}](${ref})\n`)
        )
      }

      if (poc?.length) {
        markdownOutput += `### Proof of Concept:\n\n`
        poc.forEach((p: { added_at: string | Date; url: any; source: any }) => {
          markdownOutput += `- [${p.url}](${p.url}) (Source: ${p.source}, Added: ${formatTime(p.added_at)})\n`
        })
      } else {
        markdownOutput += `\n### Proof of Concept Available: ${is_poc ? "Yes" : "No"}\n`
      }

      if (nuclei_templates) {
        const {
          created_at,
          template_issue,
          template_issue_type,
          template_path,
          template_pr,
          template_url,
          updated_at
        } = nuclei_templates
        markdownOutput += `### Nuclei Template Data\n`
        markdownOutput += `- **Created At**: ${created_at}\n`
        markdownOutput += `- **Template Issue**: [${template_issue}](${template_issue})\n`
        markdownOutput += `- **Template Issue Type**: ${template_issue_type}\n`
        markdownOutput += `- **Template Path**: ${template_path}\n`
        markdownOutput += `- **Template PR**: [${template_pr}](${template_pr})\n`
        markdownOutput += `- **Template URL**: [${template_url}](${template_url})\n`
        markdownOutput += `- **Updated At**: ${updated_at}\n\n`
      } else {
        markdownOutput += `\n### Nuclei Template Available: ${is_template ? "Yes" : "No"}\n`
      }

      // if (hackerone?.rank || hackerone?.count !== undefined) {
      //   markdownOutput += `- **HackerOne**: Rank ${hackerone.rank}, Reports ${hackerone.count}\n`;
      // }

      if (shodan?.count) {
        markdownOutput += `### Shodan Data\n`
        markdownOutput += `- **Number of Results**: ${shodan.count}\n`
        if (shodan.query?.length) {
          markdownOutput += `- **Queries**:\n`
          shodan.query.forEach(
            (query: any) => (markdownOutput += `  - \`${query}\`\n`)
          )
        }
      }

      markdownOutput += `### Other\n`

      addOptionalField("Vulnerability Status", vuln_status)
      addOptionalField("Remotely Exploitable", is_remote ? "Yes" : "No")
      addOptionalField("Open Source Software", is_oss ? "Yes" : "No")
      if (vendor_advisory)
        markdownOutput += `- **Vendor Advisory**: ${vendor_advisory}\n`
      addOptionalField("Exploited in the Wild", is_exploited ? "Yes" : "No")

      if (oss?.url) {
        markdownOutput += `- **OSS**: [${oss.url}](${oss.url})\n`
      }

      if (patch_url?.length) {
        markdownOutput += `- **Patch URL**:\n`
        patch_url.forEach((url: any) => (markdownOutput += `  - ${url}\n`))
      }

      markdownOutput += "\n"
    }
  )

  return markdownOutput
}
