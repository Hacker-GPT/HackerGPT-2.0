import { LLM } from "@/types"

const MISTRAL_PLATORM_LINK = "https://docs.mistral.ai/"

// Mistral Models (UPDATED 12/21/23) -----------------------------

export const MISTRAL_MEDIUM: LLM = {
  modelId: "mistral-medium",
  modelName: "HackerGPT 3.5",
  shortModelName: "HGPT-3.5",
  provider: "mistral",
  hostedId: "mistral-medium",
  platformLink: MISTRAL_PLATORM_LINK,
  imageInput: false
}

export const MISTRAL_LARGE: LLM = {
  modelId: "mistral-large",
  modelName: "HackerGPT 4",
  shortModelName: "HGPT-4",
  provider: "mistral",
  hostedId: "mistral-large",
  platformLink: MISTRAL_PLATORM_LINK,
  imageInput: false
}

export const MISTRAL_LLM_LIST: LLM[] = [MISTRAL_MEDIUM, MISTRAL_LARGE]
