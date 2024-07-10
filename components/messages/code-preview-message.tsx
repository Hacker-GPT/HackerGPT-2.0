import {
  FC,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState
} from "react"
import { Label } from "@/components/ui/label"
import { Button } from "@/components/ui/button"

interface CodePreviewProps {
  language: string
  value: string
}

export const CodePreview: FC<CodePreviewProps> = memo(({ language, value }) => {
  const [error, setError] = useState<string | null>(null)
  const refIframe = useRef<HTMLIFrameElement>(null)
  const refContainer = useRef<HTMLDivElement>(null)
  const [iframeHeight, setIframeHeight] = useState<number>(400)

  const scripts = useMemo(
    () => ({
      sendHeight: `
      function sendHeight() {
        const height = Math.max(document.documentElement.offsetHeight, document.documentElement.scrollHeight);
        window.parent.postMessage({ type: "resize", height }, "*");
      }
      window.addEventListener('load', sendHeight);
      new ResizeObserver(sendHeight).observe(document.body);
    `,
      errorHandling: `
      window.onerror = function(message, source, lineno, colno, error) {
        window.parent.postMessage({ type: "error", message, source, lineno, colno }, "*");
        return true;
      };
    `
    }),
    []
  )

  const iframeContent = useMemo(() => {
    const htmlContent =
      language === "html" ? value : `<script>${value}</script>`
    return `
      <!DOCTYPE html>
      <html>
        <head>
          <script>${scripts.errorHandling}</script>
        </head>
        <body>
          ${htmlContent}
          <script>${scripts.sendHeight}</script>
        </body>
      </html>
    `
  }, [language, value, scripts])

  const updateIframeHeight = useCallback((height: number) => {
    const screenHeight = window.innerHeight
    console.log(screenHeight)
    let maxHeightPercentage: number

    if (screenHeight < 600) {
      maxHeightPercentage = 0.5 // 50% for small screens
    } else if (screenHeight < 900) {
      maxHeightPercentage = 0.6 // 60% for medium screens
    } else {
      maxHeightPercentage = 0.7 // 70% for large screens
    }

    const maxHeight = Math.min(height, screenHeight * maxHeightPercentage)
    setIframeHeight(Math.max(maxHeight, 200)) // Minimum height set to 200px
  }, [])

  useEffect(() => {
    const handleMessage = (event: MessageEvent) => {
      if (event.data.type === "resize") {
        updateIframeHeight(event.data.height)
      } else if (event.data.type === "error") {
        setError(`Error: ${event.data.message} at line ${event.data.lineno}`)
      }
    }
    window.addEventListener("message", handleMessage)
    return () => window.removeEventListener("message", handleMessage)
  }, [updateIframeHeight])

  return (
    <div ref={refContainer} className="relative w-full">
      <iframe
        ref={refIframe}
        style={{ height: `${iframeHeight}px` }}
        className="w-full border-none bg-white transition-all duration-300 ease-in-out"
        srcDoc={iframeContent}
        sandbox="allow-scripts allow-popups allow-same-origin"
        title="Code Execution Result"
      />
      {error && (
        <div className="absolute bottom-0 max-h-[200px] w-full overflow-auto rounded bg-red-100 p-3 text-sm text-red-800 shadow-lg">
          <div className="flex items-center justify-between gap-1">
            <Label className="font-semibold">Console Error</Label>
            <Button
              className="text-red-800 hover:bg-red-200"
              onClick={() => {
                navigator.clipboard.writeText(error)
              }}
              title="Copy error message"
            >
              Copy
            </Button>
          </div>
          <pre className="mt-2 whitespace-pre-wrap font-mono text-xs">
            {error}
          </pre>
        </div>
      )}
    </div>
  )
})

CodePreview.displayName = "CodePreview"
