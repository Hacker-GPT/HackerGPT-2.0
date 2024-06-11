import endent from "endent"
import { filterEmptyAssistantMessages } from "@/lib/build-prompt"

export async function generateStandaloneQuestion(
  messages: any[],
  latestUserMessage: any,
  openRouterUrl: string | URL | Request,
  openRouterHeaders: any,
  selectedStandaloneQuestionModel: string | undefined,
  systemMessageContent: string
) {
  // Removed the filter for the standalone question as we already have one before this function is called
  //if (messages.length < 4 || latestUserMessage.length > 256) {
  //return latestUserMessage
  //}

  // Faster and smaller model for standalone questions for reduced latency
  const modelStandaloneQuestion = selectedStandaloneQuestionModel

  filterEmptyAssistantMessages(messages)

  let chatHistory = messages
    .slice(1, -1) // Remove the first (system prompt) and the last message (user message)
    .slice(-3) // Get the last 3 messages only (assistant, user, assistant)
    .map(msg => `${msg.role}: ${msg.content}`)
    .join("\n")

  // Compressed prompt with HyDE
  const template = endent`
    Your are HackerGPT is an expert in hacking, particularly in the areas of bug bounty, hacking, penetration testing. You are having a conversation with an user and you want to enrich your answer with some expert knowledge.
    Objective 1: Craft a standalone question for a specialist who is unfamiliar with the conversation, based on the given follow-up question and chat history. The question should:
  
    1. Emphasize relevant keywords
    2. Seek specific actions or information 
    3. Provide full context while being concise
    4. Be phrased as a clear, direct question
    5. Exclude irrelevant details
  
    Input:
    - Chat History: """${chatHistory}"""
    - Follow Up: """${latestUserMessage}"""
  
    Output:
    The rephrased standalone question to ask the specialist. Use the following format:
    <Standalone Question>{Your standalone question here}</Standalone Question>`

  const firstMessage = messages[0]
    ? messages[0]
    : { role: "system", content: `${systemMessageContent}` }

  try {
    const requestBody = {
      model: modelStandaloneQuestion,
      route: "fallback",
      messages: [
        { role: firstMessage.role, content: firstMessage.content },
        { role: "user", content: template }
      ],
      temperature: 0.1,
      max_tokens: 256
    }

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

    const standaloneQuestion = data.choices?.[0]?.message?.content?.trim()
    return standaloneQuestion
  } catch (error) {
    console.error("Error in generateStandaloneQuestion:", error)
    return latestUserMessage
  }
}
