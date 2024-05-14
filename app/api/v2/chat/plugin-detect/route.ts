import { getServerProfile } from "@/lib/server/server-chat-helpers"
import { buildFinalMessages } from "@/lib/build-prompt"
import llmConfig from "@/lib/models/llm/llm-config"
import { updateOrAddSystemMessage } from "@/lib/ai-helper"

class APIError extends Error {
  code: any
  constructor(message: string | undefined, code: any) {
    super(message)
    this.name = "APIError"
    this.code = code
  }
}

const availablePlugins = [
  {
    name: "cvemap",
    description:
      "is a tool that provides a structured and easily navigable way to explore CVEs from the command line. If user asked information about specfic CVE use this plugin."
  },
  {
    name: "subfinder",
    description:
      "subdomain discovery tool that finds and returns valid subdomains for websites. Using passive online sources, it has a simple modular architecture optimized for speed. Subfinder is built for one thing - passive subdomain enumeration, and it does that very well."
  },
  {
    name: "golinkfinder",
    description:
      "a fast and minimal Urls extractor. If user asked to extract or get urls from one single domain or url use this plugin."
  },
  {
    name: "nuclei",
    description:
      "a fast vulnerability scanner designed to probe modern applications, infrastructure, cloud platforms, and networks, aiding in the identification and mitigation of exploitable vulnerabilities."
  },
  {
    name: "katana",
    description:
      "a web crawling tool written in Golang, designed to be fast, efficient, and provide simple output."
  },
  {
    name: "httpx",
    description:
      "a fast and multi-purpose HTTP toolkit built to support running multiple probes using a public library. Probes are specific tests or checks to gather information about web servers, URLs, or other HTTP elements. Httpx is designed to maintain result reliability with an increased number of threads."
  },
  {
    name: "naabu",
    description:
      "a port scanning tool written in Go that enumerates valid ports for hosts in a fast and reliable manner. It is a really simple tool that does fast SYN/CONNECT/UDP scans on the host or list of hosts and provides all ports that return a reply."
  },
  {
    name: "alterx",
    description:
      "a fast and customizable subdomain wordlist generator that uses patterns to fit into common subdomain enumeration pipelines. Instead of relying on hardcoding patterns in the tool itself, AlterX uses a set of patterns that users can customize."
  }
]

export async function POST(request: Request) {
  const json = await request.json()
  const { payload, chatImages, selectedPlugin } = json

  try {
    const profile = await getServerProfile()

    const openrouterApiKey = profile.openrouter_api_key || ""
    let providerUrl, providerHeaders, selectedStandaloneQuestionModel

    const useOpenRouter = process.env.USE_OPENROUTER?.toLowerCase() === "true"
    if (useOpenRouter) {
      providerUrl = llmConfig.openrouter.url
      selectedStandaloneQuestionModel =
        llmConfig.models.hackerGPT_standalone_question_openrouter
      providerHeaders = {
        Authorization: `Bearer ${openrouterApiKey}`,
        "Content-Type": "application/json"
      }
    } else {
      providerUrl = llmConfig.together.url
      selectedStandaloneQuestionModel =
        llmConfig.models.hackerGPT_standalone_question_together
      providerHeaders = {
        Authorization: `Bearer ${process.env.TOGETHER_API_KEY}`,
        "Content-Type": "application/json"
      }
    }

    const messages = await buildFinalMessages(
      payload,
      profile,
      chatImages,
      selectedPlugin
    )
    const cleanedMessages = messages as any[]

    const systemMessageContent = `${llmConfig.systemPrompts.hackerGPT}`
    updateOrAddSystemMessage(cleanedMessages, systemMessageContent)

    const lastUserMessage = cleanedMessages[cleanedMessages.length - 2].content
    const detectedPlugin = await detectPlugin(
      messages,
      lastUserMessage,
      providerUrl,
      providerHeaders,
      selectedStandaloneQuestionModel
    )

    if (
      detectedPlugin === "None" ||
      !availablePlugins.map(plugin => plugin.name).includes(detectedPlugin)
    ) {
      return new Response(JSON.stringify({ plugin: "None" }), { status: 200 })
    } else {
      return new Response(JSON.stringify({ plugin: detectedPlugin }), {
        status: 200
      })
    }
  } catch (error: any) {
    if (error instanceof APIError) {
      console.error(
        `API Error - Code: ${error.code}, Message: ${error.message}`
      )
      return new Response(JSON.stringify({ error: error.message }), {
        status: error.code
      })
    } else {
      console.error(`Unexpected Error: ${error.message}`)
      return new Response(JSON.stringify({ error: "Internal server error" }), {
        status: 500
      })
    }
  }
}

async function detectPlugin(
  messages: any[],
  lastUserMessage: string,
  openRouterUrl: string | URL | Request,
  openRouterHeaders: any,
  selectedStandaloneQuestionModel: string | undefined
) {
  const modelStandaloneQuestion = selectedStandaloneQuestionModel

  const filteredMessages = messages
    .filter(msg => !(msg.role === "assistant" && msg.content === ""))
    .slice(1, -1)
    .slice(-3)
    .map(msg => `${msg.role}: ${msg.content}`)

  const pluginsInfo = availablePlugins
    .map(plugin => `- ${plugin.name}: ${plugin.description}`)
    .join("\n")

  const template = `
    You are having a conversation with a user and need to determine if the user wants to use a specific plugin for their task.
    Objective: Based on the given follow-up question and chat history, determine if the user wants to use a plugin. The available plugins are:
    ${pluginsInfo}
  
    Input:
    - Query: """${lastUserMessage}"""
  
    Output:
    Use the following guidelines to analyze the user's request:
    1. Identify if the request involves an action that can be directly executed by a plugin within the chat environment (e.g., scanning a website, finding subdomains).
    2. If the user's request relates to obtaining information that does not directly engage a plugin’s functional operation within the system (e.g., installation instructions, conceptual explanations), respond with <Plugin>None</Plugin>.
    3. If the request clearly involves an action that matches a plugin's capabilities (like performing a scan or discovery), respond with the name of the plugin, wrapped in <Plugin> tags. Ensure the plugin name is in lowercase.
    
    Use this format for the response:
    <Plugin>{None or plugin name}</Plugin>
  
    Decision Criteria:
    - For requests like 'scan a site for vulnerabilities' or 'find all subdomains of a domain', use the respective plugin capable of these actions.
    - For requests like 'how to install a plugin' or 'explain how this tool works', respond with <Plugin>None</Plugin>, as these do not require direct plugin intervention.`

  const firstMessage = messages[0]
    ? messages[0]
    : { role: "system", content: `${llmConfig.systemPrompts.hackerGPT}` }

  try {
    const requestBody = {
      model: modelStandaloneQuestion,
      route: "fallback",
      messages: [
        { role: firstMessage.role, content: firstMessage.content },
        ...filteredMessages,
        { role: "user", content: template }
      ],
      temperature: 0.1,
      max_tokens: 64
    }
    console.log(requestBody)

    const res = await fetch(openRouterUrl, {
      method: "POST",
      headers: openRouterHeaders,
      body: JSON.stringify(requestBody)
    })

    if (!res.ok) {
      const errorBody = await res.text()
      console.error("Error Response Body:", errorBody)
      throw new Error(
        `HTTP error! status: ${res.status}. Error Body: ${errorBody}`
      )
    }

    const data = await res.json()
    const aiResponse = data.choices?.[0]?.message?.content?.trim()
    console.log(aiResponse)
    const pluginMatch = aiResponse.match(/<plugin>(.*?)<\/plugin>/i)
    const detectedPlugin = pluginMatch ? pluginMatch[1].toLowerCase() : "None"

    if (!availablePlugins.map(plugin => plugin.name).includes(detectedPlugin)) {
      return "None"
    } else {
      return detectedPlugin
    }
  } catch (error) {
    console.error("Error in detectPlugin:", error)
    return "None"
  }
}
