import { ModelProvider } from "."

export type LLMID = OpenAILLMID | MistralLLMID

// OpenAI Models (UPDATED 1/29/24)
export type OpenAILLMID =
  | "gpt-4-turbo-preview" // GPT-4 Turbo
  | "gpt-4o" // GPT-4o

// Mistral Models
export type MistralLLMID =
  | "mistral-medium" // Mistral Medium
  | "mistral-large" // Mistral Large

export interface LLM {
  modelId: LLMID
  modelName: string
  provider: ModelProvider
  hostedId: string
  imageInput: boolean
  shortModelName?: string
}

export interface OpenRouterLLM extends LLM {
  maxContext: number
}
