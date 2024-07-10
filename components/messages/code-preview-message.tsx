import React, { FC, memo } from "react"
import { HTMLPreview } from './code-preview/html-preview'

interface CodePreviewProps {
  language: string
  value: string
}

export const CodePreview: FC<CodePreviewProps> = memo(({ language, value }) => {
  if (language === "html") {
    return <HTMLPreview content={value} />
  }

  return (
    <pre className="p-4 bg-gray-100 rounded overflow-auto dark:bg-gray-800 dark:text-gray-200">
      <code>{value}</code>
    </pre>
  )
})

CodePreview.displayName = "CodePreview"