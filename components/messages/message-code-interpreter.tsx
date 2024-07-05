import React, { useState } from "react"
import { MessageMarkdown } from "./message-markdown"
import { IconChevronDown, IconChevronUp, IconCode } from "@tabler/icons-react"

interface MessageCodeInterpreterProps {
  content: string
}

interface ParsedContent {
  code: string
  results: Array<{ text: string }>
  explanation: string
}

const useCodeInterpreterContent = (content: string) => {
  const [parsedContent, setParsedContent] = useState<ParsedContent | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  React.useEffect(() => {
    setIsLoading(true)
    setError(null)
    try {
      const parsed = JSON.parse(content)
      if (parsed.type === "code_interpreter") {
        setParsedContent({
          code: parsed.code || "",
          results: Array.isArray(parsed.results) ? parsed.results : [],
          explanation: parsed.explanation || ""
        })
      } else {
        throw new Error("Invalid content type")
      }
    } catch (error) {
      console.error("Error parsing code interpreter output:", error)
      setError("Failed to parse content")
    } finally {
      setIsLoading(false)
    }
  }, [content])

  return { parsedContent, isLoading, error }
}

export const MessageCodeInterpreter: React.FC<MessageCodeInterpreterProps> = ({
  content
}) => {
  const [isAnalysisOpen, setIsAnalysisOpen] = useState(true)
  const { parsedContent, isLoading, error } = useCodeInterpreterContent(content)

  if (isLoading) {
    return (
      <div className="animate-pulse">Loading code interpreter output...</div>
    )
  }

  if (error || !parsedContent) {
    return <MessageMarkdown content={content} isAssistant={true} />
  }

  const { code, results, explanation } = parsedContent

  return (
    <div>
      <div className="border-secondary overflow-hidden rounded-lg border">
        <button
          className="bg-secondary/50 hover:bg-secondary/100 flex w-full items-center justify-between p-2 transition-colors duration-200"
          onClick={() => setIsAnalysisOpen(!isAnalysisOpen)}
          aria-expanded={isAnalysisOpen}
          aria-controls="code-interpreter-content"
        >
          <div className="flex items-center">
            <IconCode size={20} className="mr-2" />
            <h4 className="font-medium">Code Interpreter Output</h4>
          </div>
          {isAnalysisOpen ? (
            <IconChevronUp size={20} />
          ) : (
            <IconChevronDown size={20} />
          )}
        </button>
        <div
          id="code-interpreter-content"
          className={`transition-all duration-300 ease-in-out ${
            isAnalysisOpen ? "max-h-[2000px] opacity-100" : "max-h-0 opacity-0"
          }`}
        >
          {code && (
            <div className="bg-secondary/25 p-4">
              <MessageMarkdown
                content={`\`\`\`python\n${code}\n\`\`\``}
                isAssistant={true}
              />
            </div>
          )}
          {results.length > 0 && (
            <div className="border-secondary border-t p-4">
              <h5 className="mb-2 font-medium">Results:</h5>
              {results.map((result, index) => (
                <MessageMarkdown
                  key={index}
                  content={`\`\`\`\n${result.text}\n\`\`\``}
                  isAssistant={true}
                />
              ))}
            </div>
          )}
        </div>
      </div>
      {explanation && (
        <div className="mt-4">
          <MessageMarkdown content={explanation} isAssistant={true} />
        </div>
      )}
    </div>
  )
}
