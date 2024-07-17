import { Button } from "@/components/ui/button"
import { useCopyToClipboard } from "@/lib/hooks/use-copy-to-clipboard"
import {
  IconCheck,
  IconCode,
  IconCopy,
  IconPlayerPlay
} from "@tabler/icons-react"
import { FC, memo, useMemo, useState } from "react"
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter"
import { oneDark } from "react-syntax-highlighter/dist/cjs/styles/prism"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { cn } from "@/lib/utils"
import { CodePreview } from "./code-preview-message"

interface MessageCodeBlockProps {
  language: string
  value: string
}

const CopyButton: FC<{ value: string; title?: string; className?: string }> =
  memo(({ value, title = "Copy to clipboard", className }) => {
    const { isCopied, copyToClipboard } = useCopyToClipboard({ timeout: 2000 })
    return (
      <Button
        title={title}
        variant="ghost"
        size="sm"
        className={cn(
          "text-xs text-white hover:bg-zinc-800 focus-visible:ring-1 focus-visible:ring-slate-700 focus-visible:ring-offset-0",
          className
        )}
        onClick={() => !isCopied && copyToClipboard(value)}
        aria-label={isCopied ? "Copied" : "Copy to clipboard"}
      >
        <span className="flex items-center space-x-1">
          {isCopied ? <IconCheck size={16} /> : <IconCopy size={16} />}
          <span className="hidden sm:inline">
            {isCopied ? "Copied!" : "Copy"}
          </span>
        </span>
      </Button>
    )
  })

CopyButton.displayName = "CopyButton"

export const MessageCodeBlock: FC<MessageCodeBlockProps> = memo(
  ({ language, value }) => {
    const [execute, setExecute] = useState(false)
    const isExecutable = useMemo(() => {
      const isHtml = language.toLowerCase() === "html"
      const lineCount = value.split("\n").length
      return isHtml && lineCount > 15
    }, [language, value])

    return (
      <div className="codeblock relative w-full bg-zinc-950 font-sans">
        <div className="sticky top-0 flex w-full items-center justify-between bg-zinc-700 px-4 text-white">
          <span className="text-xs lowercase">{language}</span>
          <div className="flex items-center space-x-1">
            {isExecutable && (
              <ToggleGroup
                onValueChange={value => setExecute(value === "execute")}
                size="sm"
                variant="default"
                className="gap-0 overflow-hidden rounded-md border border-zinc-700"
                type="single"
                value={execute ? "execute" : "code"}
              >
                {[
                  { title: "View the code", value: "code", icon: IconCode },
                  {
                    title: "Preview the code",
                    value: "execute",
                    icon: IconPlayerPlay
                  }
                ].map(({ title, value, icon: Icon }) => (
                  <ToggleGroupItem
                    key={value}
                    title={title}
                    value={value}
                    className="space-x-1 rounded-none border-none text-xs text-white"
                  >
                    <Icon size={16} stroke={1.5} />
                    <span className="hidden sm:inline">
                      {value === "code" ? "Code" : "Preview"}
                    </span>
                  </ToggleGroupItem>
                ))}
              </ToggleGroup>
            )}
            <CopyButton value={value} />
          </div>
        </div>
        {execute ? (
          <CodePreview language={language} value={value} />
        ) : (
          <SyntaxHighlighter
            language={language}
            style={oneDark}
            customStyle={{ margin: 0, background: "transparent" }}
            codeTagProps={{
              style: { fontSize: "14px", fontFamily: "var(--font-mono)" }
            }}
          >
            {value}
          </SyntaxHighlighter>
        )}
      </div>
    )
  }
)

MessageCodeBlock.displayName = "MessageCodeBlock"
