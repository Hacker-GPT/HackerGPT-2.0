import { OpenAIEmbeddings } from "@langchain/openai"
import { CohereClient } from "cohere-ai"

interface PineconeQueryConfig {
  apiKey: string
  index: string
  projectId: string
  environment: string
  namespace?: string
  temperature: number
  messageLength: { min: number; max: number }
  pineconeNamespace?: string
}

interface CohereRerankConfig {
  apiKey: string
  rerankTopK: number
  rerank: boolean
}

class RetrieverReranker {
  private embedModel: OpenAIEmbeddings
  private similarityTopK: number
  private pineconeConfig: PineconeQueryConfig
  private cohereConfig: CohereRerankConfig

  constructor(
    openAIApiKey: string | undefined,
    pineconeConfig: PineconeQueryConfig,
    cohereConfig: CohereRerankConfig,
    similarityTopK: number
  ) {
    if (!openAIApiKey) {
      throw new Error("OpenAI API key is required.")
    }
    // Ensure all required config properties are initialized.
    if (
      !pineconeConfig.apiKey ||
      !pineconeConfig.index ||
      !pineconeConfig.projectId ||
      !pineconeConfig.environment
    ) {
      throw new Error("Pinecone configuration is missing required properties.")
    }

    this.embedModel = new OpenAIEmbeddings({
      openAIApiKey: openAIApiKey,
      modelName: "text-embedding-3-large",
      dimensions: 3072
    })
    this.cohereConfig = cohereConfig
    this.pineconeConfig = pineconeConfig
    this.similarityTopK = similarityTopK
  }

  async retrieve(question: string): Promise<string> {
    const queryEmbedding = await this.embedModel.embedQuery(question)
    const PINECONE_QUERY_URL = `https://${this.pineconeConfig.index}-${this.pineconeConfig.projectId}.svc.${this.pineconeConfig.environment}.pinecone.io/query`
    const requestBody = {
      vector: queryEmbedding,
      topK: this.cohereConfig.rerank
        ? this.cohereConfig.rerankTopK
        : this.similarityTopK * 2, // Double the topK to account for filtering
      includeMetadata: true,
      namespace: this.pineconeConfig.namespace
    }

    try {
      const response = await fetch(PINECONE_QUERY_URL, {
        method: "POST",
        headers: {
          "Api-Key": this.pineconeConfig.apiKey,
          "Content-Type": "application/json"
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      // Filter out matches with scores below 0.45
      const highScoreMatches = data.matches.filter((match: any) => {
        if (match.score <= 0.45) return false
        const nodeContent = JSON.parse(match.metadata._node_content)
        const textContent = nodeContent.text
        return textContent.length >= 70 && textContent.length <= 5000
      })

      // Remove duplicates based on nodeContent.text
      const uniqueMatches = highScoreMatches.reduce(
        (acc: any[], match: any) => {
          const nodeContent = JSON.parse(match.metadata._node_content)
          const textContent = nodeContent.text
          if (
            !acc.find(
              m => JSON.parse(m.metadata._node_content).text === textContent
            )
          ) {
            acc.push(match)
          }
          return acc
        },
        []
      )

      if (this.cohereConfig.rerank) {
        const cohere = new CohereClient({
          token: this.cohereConfig.apiKey
        })
        const uniqueMatchesText = uniqueMatches.map((match: any) => {
          const nodeContent = JSON.parse(match.metadata._node_content)
          return nodeContent.text
        })

        const isolatedQuestion =
          question
            .match(/<Standalone Question>(.*?)<\/Standalone Question>/)?.[1]
            .trim() || question

        // Rerank the chunks using cohere
        const rerankedMatches = await cohere.rerank({
          model: "rerank-english-v2.0",
          documents: uniqueMatchesText.map((d: any) => ({
            text: d
          })),
          query: isolatedQuestion,
          topN: this.similarityTopK
        })

        const cohereFormattedResults = rerankedMatches.results
          .map((result: any, index: number) => {
            return `[CONTEXT ${index}]:\n${uniqueMatchesText[result.index]}\n[END CONTEXT ${index}]\n\n`
          })
          .join("")

        return cohereFormattedResults.length > 0
          ? cohereFormattedResults
          : "None"
      }

      // Map the filtered and unique matches to the specified format
      let formattedResults = uniqueMatches
        .slice(0, this.similarityTopK)
        .map((match: any, index: number) => {
          // Parse the _node_content string to access its properties
          const nodeContent = JSON.parse(match.metadata._node_content)
          // Extract the `text` property
          const textContent = nodeContent.text

          // Format the `text` content within the specified structure
          return `[CONTEXT ${index}]:\n${textContent}\n[END CONTEXT ${index}]\n\n`
        })
        .join("")

      // Log the question, data, and formatted results for debugging
      // console.log({
      //   question,
      //   data: data.matches.map((d: any) => ({
      //     id: d.id,
      //     file_name: d.metadata.file_name,
      //     score: d.score,
      //     text: JSON.parse(d.metadata._node_content).text.slice(0, 100) + "..."
      //   })),
      //   formattedResults
      // })
      return formattedResults.length > 0 ? formattedResults : "None"
    } catch (error) {
      console.error(`Error querying Pinecone/Cohere: ${error}`)
      return "Error retrieving data"
    }
  }
}

export default RetrieverReranker
