import { ChatbotUIContext } from "@/context/context"
import {
  type UIEventHandler,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState
} from "react"

export const useScroll = () => {
  const { isGenerating, chatMessages } = useContext(ChatbotUIContext)

  const messagesStartRef = useRef<HTMLDivElement>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const isAutoScrolling = useRef(false)

  const [isAtTop, setIsAtTop] = useState(false)
  const [isAtBottom, setIsAtBottom] = useState(true)
  const [userScrolled, setUserScrolled] = useState(false)
  const [isOverflowing, setIsOverflowing] = useState(false)

  useEffect(() => {
    // setUserScrolled(false)
    console.log("isGenerating", isGenerating)
    if (!isGenerating && userScrolled) {
      setUserScrolled(false)
    }
  }, [isGenerating])

  useEffect(() => {
    if (isGenerating && !userScrolled) {
      scrollToBottom()
    }
  }, [chatMessages])

  const handleScroll: UIEventHandler<HTMLDivElement> = useCallback(e => {
    const target = e.target as HTMLDivElement
    const bottom =
      Math.abs(target.scrollHeight - target.scrollTop - target.clientHeight) <
      250
    setTimeout(() => {
      setIsAtBottom(bottom)
    }, 300)

    const top = target.scrollTop < 250
    setIsAtTop(top)

    console.log(
      "bottom",
      bottom,
      "isAutoScrolling",
      isAutoScrolling.current,
      "userScrolled",
      userScrolled
    )

    // Set userScrolled based on scroll position and auto-scrolling status
    if (!isAutoScrolling.current) {
      if (!bottom || top) {
        setUserScrolled(true)
      }
    } else if (bottom) {
      setUserScrolled(false) // Reset when auto-scrolling to bottom completes
    }

    const isOverflow = target.scrollHeight > target.clientHeight
    setIsOverflowing(isOverflow)
  }, [])

  const scrollToTop = useCallback(() => {
    if (messagesStartRef.current) {
      messagesStartRef.current.scrollIntoView({ behavior: "smooth" })
    }
  }, [])

  const scrollToBottom = useCallback(() => {
    isAutoScrolling.current = true

    setTimeout(() => {
      if (messagesEndRef.current && !userScrolled) {
        messagesEndRef.current.scrollIntoView({ behavior: "instant" })
      }

      isAutoScrolling.current = false
    }, 200)
  }, [])

  return {
    messagesStartRef,
    messagesEndRef,
    isAtTop,
    isAtBottom,
    userScrolled,
    isOverflowing,
    handleScroll,
    scrollToTop,
    scrollToBottom,
    setIsAtBottom
  }
}
