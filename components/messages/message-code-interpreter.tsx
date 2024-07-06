import React, { useState, useMemo } from "react"
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
  const { code, results, explanation, otherContent } = useMemo(
    () => parseCodeInterpreterContent(content),
    [content]
  )

  const hasCodeOutput = code || results.length > 0 || explanation

  return (
    <div>
      {otherContent && (
        <MessageMarkdown content={otherContent} isAssistant={true} />
      )}
      {hasCodeOutput && (
        <div className="border-secondary my-4 overflow-hidden rounded-lg border">
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
          {isAnalysisOpen && (
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
          )}
        </div>
      )}
      {explanation && (
        <MessageMarkdown content={explanation} isAssistant={true} />
      )}
    </div>
  )
}

const parseCodeInterpreterContent = (content: string): ParsedContent => {
  const lines = content.split("\n")
  const newContent: ParsedContent = {
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
        case "code_interpreter_input":
          newContent.code = parsed.content
          break
        case "code_interpreter_output":
          newContent.results = parsed.content.map((item: any) => ({
            text: typeof item.text === "object" ? item.text.text : item.text
          }))
          break
        case "explanation":
          currentSection = "explanation"
          newContent.explanation += parsed.content + "\n"
          break
        default:
          newContent.otherContent += line + "\n"
      }
    } catch {
      // If it's not JSON, add it to the current section
      newContent[currentSection] += line + "\n"
    }
  })

  // Trim trailing newlines
  newContent.explanation = newContent.explanation.trim()
  newContent.otherContent = newContent.otherContent.trim()

  return newContent
}
