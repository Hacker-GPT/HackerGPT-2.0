import { LLM } from "@/types"

export const HGPT3_5: LLM = {
  modelId: "mistral-medium",
  modelName: "HackerGPT 3.5",
  shortModelName: "HGPT-3.5",
  provider: "mistral",
  hostedId: "mistral-medium",
  imageInput: false
}

export const HGPT4: LLM = {
  modelId: "mistral-large",
  modelName: "HackerGPT 4",
  shortModelName: "HGPT-4",
  provider: "mistral",
  hostedId: "mistral-large",
  imageInput: false
}

export const HACKERAI_LLM_LIST: LLM[] = [HGPT3_5, HGPT4]
