import { LLM } from "@/types"

const OPENAI_PLATORM_LINK = "https://platform.openai.com/docs/overview"

// OpenAI Models (UPDATED 1/25/24) -----------------------------

// GPT-4 Turbo (UPDATED 5/15/24)
export const GPT4: LLM = {
  modelId: "gpt-4-turbo-preview", // Not a good idea to change as it could be stored in browser and it's in the DB, carefully change this if required.
  modelName: "GPT-4o",
  provider: "openai",
  hostedId: "gpt-4o",
  platformLink: OPENAI_PLATORM_LINK,
  imageInput: true
}

export const OPENAI_LLM_LIST: LLM[] = [GPT4]
