import React, { useState, useEffect } from "react"
import { MessageMarkdown } from "./message-markdown"
import { IconChevronDown, IconChevronUp, IconCode } from "@tabler/icons-react"

interface MessageCodeInterpreterProps {
  content: string
}

interface ParsedContent {
  code: string
  results: Array<{ text: string }>
  explanation: string
  otherContent: string
}

export const MessageCodeInterpreter: React.FC<MessageCodeInterpreterProps> = ({
  content
}) => {
  const [isAnalysisOpen, setIsAnalysisOpen] = useState(true)
  const { code, results, explanation, otherContent } =
    useCodeInterpreterContent(content)

  return (
    <div>
      {otherContent && (
        <MessageMarkdown content={otherContent} isAssistant={true} />
      )}
      {(code || results.length > 0 || explanation) && (
        <div className="border-secondary my-3 overflow-hidden rounded-lg border">
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
              isAnalysisOpen
                ? "max-h-[2000px] opacity-100"
                : "max-h-0 opacity-0"
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
      )}
      {explanation && (
        <MessageMarkdown content={explanation} isAssistant={true} />
      )}
    </div>
  )
}

const useCodeInterpreterContent = (content: string) => {
  const [parsedContent, setParsedContent] = useState<ParsedContent>({
    code: "",
    results: [],
    explanation: "",
    otherContent: ""
  })

  useEffect(() => {
    const lines = content.split("\n")
    let newContent: ParsedContent & {
      [key: string]: string | Array<{ text: string }>
    } = {
      code: "",
      results: [],
      explanation: "",
      otherContent: ""
    }
    let currentSection: "explanation" | "otherContent" = "otherContent"

    lines.forEach(line => {
      try {
        const parsed = JSON.parse(line)
        switch (parsed.type) {
          case "code":
            newContent.code = parsed.content
            break
          case "results":
            newContent.results = parsed.content.map((item: any) => ({
              text: typeof item.text === "object" ? item.text.text : item.text
            }))
            break
          case "explanation":
            currentSection = "explanation"
            break
        }
      } catch {
        // If it's not JSON, add it to the current section
        newContent[currentSection] += line + "\n"
      }
    })

    // Trim only trailing newlines
    newContent.explanation = newContent.explanation.replace(/\n+$/, "")
    newContent.otherContent = newContent.otherContent.replace(/\n+$/, "")

    setParsedContent(newContent as ParsedContent)
  }, [content])

  return parsedContent
}
