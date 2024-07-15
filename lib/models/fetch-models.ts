import { LLM } from "@/types"
import { LLM_LIST_MAP } from "./llm/llm-list"

export const fetchHostedModels = async () => {
  try {
    const providers = ["openai", "mistral"] as const

    // Hardcoded isUsingEnvKeyMap data with type assertion
    const isUsingEnvKeyMap = {
      openai: true,
      mistral: true,
      openrouter: true,
      openai_organization_id: false
    } as const

    let modelsToAdd: LLM[] = []

    for (const provider of providers) {
      if (isUsingEnvKeyMap[provider]) {
        const models = LLM_LIST_MAP[provider]
        if (Array.isArray(models)) {
          modelsToAdd.push(...models)
        }
      }
    }

    return {
      envKeyMap: isUsingEnvKeyMap,
      hostedModels: modelsToAdd
    }
  } catch (error) {
    console.warn("Error fetching hosted models:", error)
    return { envKeyMap: {}, hostedModels: [] }
  }
}
