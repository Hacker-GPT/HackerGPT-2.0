import { checkApiKey, getServerProfile } from "@/lib/server/server-chat-helpers"
import { ChatSettings } from "@/types"
import { OpenAIStream, StreamingTextResponse } from "ai"
import { ServerRuntime } from "next"
import { combinedEnglishAndCybersecurityWords } from "@/lib/hackergpt-wordlist"
import { OpenAIEmbeddings } from "langchain/embeddings/openai"

import {
  replaceWordsInLastUserMessage,
  wordReplacements
} from "@/lib/word-replacer"
import {
  updateOrAddSystemMessage,
  Message,
  APIError
} from "@/app/api/chat/custom/route"

export const runtime: ServerRuntime = "edge"

export async function POST(request: Request) {
  const json = await request.json()
  const { chatSettings, messages } = json as {
    chatSettings: ChatSettings
    messages: any[]
  }

  try {
    const profile = await getServerProfile()

    checkApiKey(profile.openrouter_api_key, "OpenRouter")

    const openrouterApiKey = profile.openrouter_api_key || ""

    const HACKERGPT_SYSTEM_PROMPT = process.env.SECRET_HACKERGPT_SYSTEM_PROMPT

    const USE_PINECONE = process.env.USE_PINECONE === "TRUE"
    const PINECONE_API_KEY = process.env.SECRET_PINECONE_API_KEY
    const PINECONE_ENVIRONMENT = process.env.SECRET_PINECONE_ENVIRONMENT
    const PINECONE_NAMESPACE = process.env.SECRET_PINECONE_NAMESPACE
    const PINECONE_INDEX = process.env.SECRET_PINECONE_INDEX
    const PINECONE_PROJECT_ID = process.env.SECRET_PINECONE_PROJECT_ID
    const PINECONE_SYSTEM_PROMPT = process.env.PINECONE_SYSTEM_PROMPT
    const TRANSLATE_OPENROUTER_MODEL = process.env.TRANSLATE_OPENROUTER_MODEL

    let modelTemperature = 0.4
    const pineconeTemperature = 0.7

    const openRouterUrl = `https://openrouter.ai/api/v1/chat/completions`
    const openRouterHeaders = {
      Authorization: `Bearer ${openrouterApiKey}`,
      "HTTP-Referer": "https://www.hackergpt.co",
      "X-Title": "HackerGPT",
      "Content-Type": "application/json"
    }

    let cleanedMessages = []
    const MESSAGE_USAGE_CAP_WARNING = "Hold On! You've Hit Your Usage Cap."
    const MESSAGE_SIGN_IN_WARNING = "Whoa, hold on a sec!"
    const MESSAGE_TOOL_USAGE_CAP_WARNING = "⏰ You can use the tool again in"
    const FREE_MESSAGES_WARNING = "We apologize for any inconvenience, but"

    const MIN_LAST_MESSAGE_LENGTH = parseInt(
      process.env.MIN_LAST_MESSAGE_LENGTH || "50",
      10
    )
    const MAX_LAST_MESSAGE_LENGTH = parseInt(
      process.env.MAX_LAST_MESSAGE_LENGTH || "1000",
      10
    )

    for (let i = 0; i < messages.length - 1; i++) {
      const message = messages[i]
      const nextMessage = messages[i + 1]

      if (
        !message ||
        !nextMessage ||
        typeof message.role === "undefined" ||
        typeof nextMessage.role === "undefined"
      ) {
        console.error(
          "One of the messages is undefined or does not have a role property"
        )
        continue
      }

      if (
        nextMessage.role === "assistant" &&
        nextMessage.content.includes(MESSAGE_USAGE_CAP_WARNING)
      ) {
        if (message.role === "user") {
          i++
          continue
        }
      } else if (
        nextMessage.role === "assistant" &&
        nextMessage.content.includes(MESSAGE_SIGN_IN_WARNING)
      ) {
        if (message.role === "user") {
          i++
          continue
        }
      } else if (
        nextMessage.role === "assistant" &&
        nextMessage.content.includes(MESSAGE_TOOL_USAGE_CAP_WARNING)
      ) {
        if (message.role === "user") {
          i++
          continue
        }
      } else if (
        nextMessage.role === "assistant" &&
        nextMessage.content.includes(FREE_MESSAGES_WARNING)
      ) {
        if (message.role === "user") {
          i++
          continue
        }
      }
      // Skip consecutive user messages
      else if (nextMessage.role === "user" && message.role === "user") {
        continue
      } else {
        cleanedMessages.push(message)
      }
    }

    if (
      messages[messages.length - 1].role === "user" &&
      !messages[messages.length - 1].content.includes(
        MESSAGE_USAGE_CAP_WARNING
      ) &&
      !messages[messages.length - 1].content.includes(
        MESSAGE_SIGN_IN_WARNING
      ) &&
      !messages[messages.length - 1].content.includes(
        MESSAGE_TOOL_USAGE_CAP_WARNING
      ) &&
      !messages[messages.length - 1].content.includes(FREE_MESSAGES_WARNING) &&
      (cleanedMessages.length === 0 ||
        cleanedMessages[cleanedMessages.length - 1].role !== "user")
    ) {
      cleanedMessages.push(messages[messages.length - 1])
    }

    if (
      cleanedMessages.length % 2 === 0 &&
      cleanedMessages[0]?.role === "assistant"
    ) {
      cleanedMessages.shift()
    }

    const queryPineconeVectorStore = async (query: string) => {
      const embeddingsInstance = new OpenAIEmbeddings({
        openAIApiKey: process.env.OPENAI_API_KEY
      })

      const queryEmbedding = await embeddingsInstance.embedQuery(query)

      const PINECONE_QUERY_URL = `https://${PINECONE_INDEX}-${PINECONE_PROJECT_ID}.svc.${PINECONE_ENVIRONMENT}.pinecone.io/query`

      const requestBody = {
        topK: 5,
        vector: queryEmbedding,
        includeMetadata: true,
        namespace: `${PINECONE_NAMESPACE}`
      }

      try {
        const response = await fetch(PINECONE_QUERY_URL, {
          method: "POST",
          headers: {
            "Api-Key": `${PINECONE_API_KEY}`,
            "Content-Type": "application/json"
          },
          body: JSON.stringify(requestBody)
        })

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        const data = await response.json()
        const matches = data.matches || []

        const minimumContextCount = 3
        if (matches.length < minimumContextCount) {
          return "None"
        }

        // Filter matches with score > 0.80 and metadata.text length >= 250
        const filteredMatches = matches.filter(
          (match: { score: number; metadata: { text: string } }) =>
            match.score > 0.8 && match.metadata.text.length >= 250
        )

        if (filteredMatches.length > 0) {
          let formattedResults = filteredMatches
            .map((match: { metadata: { text: string } }, index: any) => {
              const contextText = match.metadata?.text || ""
              return `[CONTEXT ${index}]:\n${contextText}\n[END CONTEXT ${index}]\n\n`
            })
            .join("")

          while (formattedResults.length > 7500) {
            let lastContextIndex = formattedResults.lastIndexOf("[CONTEXT ")
            if (lastContextIndex === -1) {
              break
            }
            formattedResults = formattedResults
              .substring(0, lastContextIndex)
              .trim()
          }

          return formattedResults || "None"
        } else {
          return "None"
        }
      } catch (error) {
        console.error(`Error querying Pinecone: ${error}`)
        return "None"
      }
    }

    let systemMessage: Message = {
      role: "system",
      content: `${HACKERGPT_SYSTEM_PROMPT}`
    }

    const translateToEnglish = async (text: any) => {
      const requestBody = {
        model: [`${TRANSLATE_OPENROUTER_MODEL}`],
        messages: [
          {
            role: "system",
            content:
              "You are a translation AI. " +
              "As a translation AI, your primary objective is to translate user-submitted text into English with high accuracy. " +
              "Focus on providing translations that are clear and direct. " +
              "Avoid adding any additional comments or information. " +
              "If the user's query is already in English, simply return the query as it is. " +
              "Your role is exclusively to translate; do not deviate from this task or engage in answering user queries."
          },
          {
            role: "user",
            content:
              "Translate the provided text into English. " +
              "Aim for an accurate and succinct translation into English. " +
              "The translation should accurately reflect the original text's meaning and context, without any supplementary comments, opinions, or extraneous information. " +
              "Refrain from engaging in discussions or asking for interpretations. " +
              "Avoid engaging in discussions or providing interpretations beyond the translation." +
              "Translate: " +
              text
          }
        ],
        temperature: 0.1,
        route: "fallback"
      }

      try {
        const request = await fetch(openRouterUrl, {
          method: "POST",
          headers: openRouterHeaders,
          body: JSON.stringify(requestBody)
        })

        if (!request.ok) {
          const response = await request.json()
          console.error("Error Code:", response.error?.code)
          console.error("Error Message:", response.error?.message)
          throw new Error(`OpenRouter error: ${response.error?.message}`)
        }

        const data = await request.json()
        return data.choices[0].message.content
      } catch (error) {
        console.error(`Error during translation: ${error}`)
        return ""
      }
    }

    const isEnglish = async (text: string, threshold = 20) => {
      combinedEnglishAndCybersecurityWords

      const words = text.toLowerCase().split(/\s+/)
      const relevantWordCount = words.filter(word =>
        combinedEnglishAndCybersecurityWords.has(word)
      ).length
      return relevantWordCount / words.length >= threshold / 100
    }

    function extractAssistantSnippet(message: string, length: number) {
      // Identify key sentences or phrases in the assistant's message
      // For simplicity, we're still using a quarter way through for now
      const startOffset = Math.floor(message.length * 0.25)
      const endOffset = startOffset + length
      return message.substring(startOffset, Math.min(endOffset, message.length))
    }

    function findNaturalBreakpoint(message: string | string[], maxLength: any) {
      // Find a natural breakpoint like the end of a sentence
      let breakpoint = message.lastIndexOf(". ", maxLength)
      return breakpoint === -1 ? maxLength : breakpoint + 1
    }

    function isMessageRelevant(message: string | any[], keywords: any[]) {
      // Basic keyword matching for relevance - can be replaced with more advanced NLP
      return keywords.some(keyword => message.includes(keyword))
    }

    if (
      USE_PINECONE &&
      cleanedMessages.length > 0 &&
      cleanedMessages[cleanedMessages.length - 1].role === "user" &&
      cleanedMessages[cleanedMessages.length - 1].content.length >
        MIN_LAST_MESSAGE_LENGTH
    ) {
      const MAX_LENGTH_FOR_INDIVIDUAL_MESSAGE = 300
      const MAX_TOTAL_LENGTH = MAX_LAST_MESSAGE_LENGTH
      const ASSISTANT_SNIPPET_LENGTH = 300
      let combinedContent = ""
      // Add a snippet of the latest assistant message for context
      for (let i = cleanedMessages.length - 2; i >= 0; i--) {
        if (cleanedMessages[i].role === "assistant") {
          combinedContent =
            extractAssistantSnippet(
              cleanedMessages[i].content,
              ASSISTANT_SNIPPET_LENGTH
            ) + " "
          break
        }
      }

      let latestUserMessage =
        cleanedMessages[cleanedMessages.length - 1].content
      let keywords = latestUserMessage.split(" ").slice(0, 5) // Example: Taking first 5 words as keywords

      // Process user messages starting from the most recent
      for (let i = cleanedMessages.length - 1; i >= 0; i--) {
        if (cleanedMessages[i].role === "user") {
          let messageContent = cleanedMessages[i].content

          // Check for relevance of the message
          if (
            !isMessageRelevant(messageContent, keywords) &&
            i != cleanedMessages.length - 1
          ) {
            continue
          }

          if (
            combinedContent.length + messageContent.length >
            MAX_TOTAL_LENGTH
          ) {
            let remainingLength = MAX_TOTAL_LENGTH - combinedContent.length
            let breakpoint = findNaturalBreakpoint(
              messageContent,
              remainingLength
            )
            combinedContent =
              messageContent.substring(0, breakpoint) + " " + combinedContent
            break
          } else if (
            messageContent.length > MAX_LENGTH_FOR_INDIVIDUAL_MESSAGE
          ) {
            let breakpoint = findNaturalBreakpoint(
              messageContent,
              MAX_LENGTH_FOR_INDIVIDUAL_MESSAGE
            )
            combinedContent =
              messageContent.substring(0, breakpoint) + " " + combinedContent
          } else {
            combinedContent = messageContent + " " + combinedContent
          }
        }
      }

      if (!(await isEnglish(combinedContent))) {
        combinedContent = await translateToEnglish(combinedContent)
      }

      const pineconeResults = await queryPineconeVectorStore(
        combinedContent.trim()
      )

      if (pineconeResults !== "None") {
        modelTemperature = pineconeTemperature

        systemMessage.content =
          `${HACKERGPT_SYSTEM_PROMPT} ` +
          `${PINECONE_SYSTEM_PROMPT} ` +
          `Context:\n ${pineconeResults}`
      }
    }

    if (cleanedMessages[0]?.role !== "system") {
      cleanedMessages.unshift(systemMessage)
    } else {
      const systemMessageContent = `${HACKERGPT_SYSTEM_PROMPT}`
      updateOrAddSystemMessage(messages, systemMessageContent)
    }

    replaceWordsInLastUserMessage(messages, wordReplacements)

    const model1 = "nousresearch/nous-hermes-2-mixtral-8x7b-dpo"
    const model2 = `${process.env.HACKERGPT_OPENROUTER_MODEL}`
    const selectedModel = Math.random() < 0.66 ? model1 : model2

    const requestBody = {
      model: selectedModel,
      route: "fallback",
      messages: cleanedMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      })),
      temperature: modelTemperature,
      max_tokens: 1024,
      stream: true
    }

    try {
      const res = await fetch(openRouterUrl, {
        method: "POST",
        headers: openRouterHeaders,
        body: JSON.stringify(requestBody)
      })

      if (!res.ok) {
        const result = await res.json()
        let errorMessage = result.error?.message || "An unknown error occurred"

        switch (res.status) {
          case 400:
            throw new APIError(`Bad Request: ${errorMessage}`, 400)
          case 401:
            throw new APIError(`Invalid Credentials: ${errorMessage}`, 401)
          case 402:
            throw new APIError(`Out of Credits: ${errorMessage}`, 402)
          case 403:
            throw new APIError(`Moderation Required: ${errorMessage}`, 403)
          case 408:
            throw new APIError(`Request Timeout: ${errorMessage}`, 408)
          case 429:
            throw new APIError(`Rate Limited: ${errorMessage}`, 429)
          case 502:
            throw new APIError(`Service Unavailable: ${errorMessage}`, 502)
          default:
            throw new APIError(`HTTP Error: ${errorMessage}`, res.status)
        }
      }

      if (!res.body) {
        throw new Error("Response body is null")
      }

      // Convert the response into a friendly text-stream.
      const stream = OpenAIStream(res)

      // Respond with the stream
      return new StreamingTextResponse(stream)
    } catch (error) {
      if (error instanceof APIError) {
        console.error(
          `API Error - Code: ${error.code}, Message: ${error.message}`
        )
      } else if (error instanceof Error) {
        console.error(`Unexpected Error: ${error.message}`)
      } else {
        console.error(`An unknown error occurred: ${error}`)
      }
    }
  } catch (error: any) {
    let errorMessage = error.message || "An unexpected error occurred"
    const errorCode = error.status || 500

    if (errorMessage.toLowerCase().includes("api key not found")) {
      errorMessage =
        "Mistral API Key not found. Please set it in your profile settings."
    } else if (errorCode === 401) {
      errorMessage =
        "Mistral API Key is incorrect. Please fix it in your profile settings."
    }

    return new Response(JSON.stringify({ message: errorMessage }), {
      status: errorCode
    })
  }
}
