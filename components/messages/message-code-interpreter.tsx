import React, { useState, useMemo } from "react"
import { MessageMarkdown } from "./message-markdown"
import {
  IconChevronDown,
  IconChevronUp,
  IconCircleCheck,
  IconExclamationCircle,
  IconLoader2
} from "@tabler/icons-react"
import { PluginID } from "@/types/plugins"
import { MessageTooLong } from "./message-too-long"

interface MessageCodeInterpreterProps {
  content: string
  messageId?: string
}

type InterpreterStatus = "idle" | "running" | "finished" | "error"

interface ParsedContent {
  code: string
  results: Array<{ text: string }>
  otherContent: string
  error: string | null
}

export const MessageCodeInterpreter: React.FC<MessageCodeInterpreterProps> = ({
  content,
  messageId
}) => {
  const [isAnalysisOpen, setIsAnalysisOpen] = useState(true)
  const [interpreterStatus, setInterpreterStatus] =
    useState<InterpreterStatus>("idle")
  const { code, results, otherContent, error } = useMemo(
    () => parseCodeInterpreterContent(content, setInterpreterStatus),
    [content]
  )

  const hasCodeOutput = code || results.length > 0 || error

  const getStatusIndicator = () => {
    switch (interpreterStatus) {
      case "running":
        return <IconLoader2 size={20} className="animate-spin" />
      case "finished":
        return <IconCircleCheck size={20} className="" />
      case "error":
        return <IconExclamationCircle size={20} className="" />
      default:
        return null
    }
  }

  const renderContent = (content: string, type: string) => {
    const contentLength = content.length
    if (contentLength > 2000) {
      return (
        <MessageTooLong
          content={content}
          plugin={PluginID.CODE_INTERPRETER}
          id={messageId || ""}
        />
      )
    }
    return (
      <MessageMarkdown
        content={`\`\`\`${type}\n${content}\n\`\`\``}
        isAssistant={true}
      />
    )
  }

  return (
    <div>
      {otherContent && (
        <MessageMarkdown content={otherContent} isAssistant={true} />
      )}
      {hasCodeOutput && (
        <div className="overflow-hidden">
          <button
            className="flex w-full items-center justify-between transition-colors duration-200"
            onClick={() => setIsAnalysisOpen(!isAnalysisOpen)}
            aria-expanded={isAnalysisOpen}
            aria-controls="code-interpreter-content"
          >
            <div className="flex items-center">
              <div>{getStatusIndicator()}</div>
              <h4 className="mx-2 font-medium">Code Interpreter</h4>
              {isAnalysisOpen ? (
                <IconChevronUp size={20} />
              ) : (
                <IconChevronDown size={20} />
              )}
            </div>
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
                <div className="pt-4">
                  <MessageMarkdown
                    content={`\`\`\`python\n${code}\n\`\`\``}
                    isAssistant={true}
                  />
                </div>
              )}
              {error ? (
                <div>{renderContent(error, "error")}</div>
              ) : (
                results.length > 0 && (
                  <div>
                    {results.map((result, index) => (
                      <React.Fragment key={index}>
                        {renderContent(result.text, "result")}
                      </React.Fragment>
                    ))}
                  </div>
                )
              )}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

const parseCodeInterpreterContent = (
  content: string,
  setInterpreterStatus: React.Dispatch<React.SetStateAction<InterpreterStatus>>
): ParsedContent => {
  const newContent: ParsedContent = {
    code: "",
    results: [],
    otherContent: "",
    error: null
  }

  const parts = content.split(/(?<=})(?={")/g)
  let hasOutput = false

  parts.forEach(part => {
    try {
      const parsed = JSON.parse(part)
      if (parsed.code) {
        newContent.code = parsed.code
        setInterpreterStatus("running")
      } else if (parsed.results) {
        newContent.results.push({ text: parsed.results })
        hasOutput = true
      } else if (parsed.runtimeError) {
        newContent.error = parsed.runtimeError
        hasOutput = true
      } else {
        newContent.otherContent += part + "\n"
      }
    } catch {
      newContent.otherContent += part + "\n"
    }
  })

  newContent.otherContent = newContent.otherContent.trim()

  if (hasOutput || (newContent.code && !newContent.error)) {
    setInterpreterStatus("finished")
  }

  return newContent
}
