const KnowledgeCutOFFOpenAI = "Knowledge cutoff: 2023-10"
const KnowledgeCutOFFMeta = "Knowledge cutoff: 2023-12"
const options: Intl.DateTimeFormatOptions = {
  year: "numeric",
  month: "long",
  day: "numeric"
}
const currentDate = `Current date: ${new Date().toLocaleDateString("en-US", options)}`

const llmConfig = {
  openrouter: {
    url: `https://openrouter.ai/api/v1/chat/completions`,
    providerRouting: {
      order: [`${process.env.OPENROUTER_FIRST_PROVIDER}`]
    },
    apiKey: process.env.OPENROUTER_API_KEY
  },
  together: {
    url: `https://api.together.xyz/v1/chat/completions`,
    apiKey: process.env.TOGETHER_API_KEY
  },
  openai: {
    url: "https://api.openai.com/v1/chat/completions",
    apiKey: process.env.OPENAI_API_KEY
  },
  systemPrompts: {
    pentestgpt: `${process.env.SECRET_HACKERGPT_SYSTEM_PROMPT}\n${KnowledgeCutOFFMeta}\n${currentDate}`,
    pentestgptCurrentDateOnly: `${process.env.SECRET_HACKERGPT_SYSTEM_PROMPT}\n${currentDate}`,
    openai: `${process.env.SECRET_OPENAI_SYSTEM_PROMPT}\n${KnowledgeCutOFFOpenAI}\n${currentDate}`,
    openaiCurrentDateOnly: `${process.env.SECRET_OPENAI_SYSTEM_PROMPT}\n${currentDate}`,
    RAG: `${process.env.SECRET_HACKERGPT_SYSTEM_PROMPT} ${process.env.RAG_SYSTEM_PROMPT}\n${currentDate}`
  },
  models: {
    pentestgpt_default_openrouter:
      process.env.OPENROUTER_HACKERGPT_DEFUALT_MODEL,
    pentestgpt_default_together: process.env.TOGETHER_HACKERGPT_DEFAULT_MODEL,
    pentestgpt_RAG_openrouter: process.env.OPENROUTER_HACKERGPT_RAG_MODEL,
    pentestgpt_RAG_together: process.env.TOGETHER_HACKERGPT_RAG_MODEL,
    pentestgpt_standalone_question_openrouter:
      process.env.OPENROUTER_STANDALONE_QUESTION_MODEL,
    pentestgpt_standalone_question_together:
      process.env.TOGETHER_STANDALONE_QUESTION_MODEL,
    pentestgpt_pro_openrouter: process.env.OPENROUTER_HACKERGPT_PRO_MODEL,
    pentestgpt_pro_together: process.env.TOGETHER_HACKERGPT_PRO_MODEL
  },
  useOpenRouter:
    (process.env.USE_OPENROUTER?.toLowerCase() || "true") === "true",
  hackerRAG: {
    enabled:
      (process.env.HACKER_RAG_ENABLED?.toLowerCase() || "false") === "true",
    endpoint: process.env.HACKER_RAG_ENDPOINT,
    getDataEndpoint: process.env.HACKER_RAG_GET_DATA_ENDPOINT,
    apiKey: process.env.HACKER_RAG_API_KEY,
    messageLength: {
      min: parseInt(process.env.MIN_LAST_MESSAGE_LENGTH || "25", 10),
      max: parseInt(process.env.MAX_LAST_MESSAGE_LENGTH || "1000", 10)
    }
  }
}

export default llmConfig
