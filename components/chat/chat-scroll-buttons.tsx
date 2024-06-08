import { IconArrowDown } from "@tabler/icons-react"
import { FC } from "react"

interface ChatScrollButtonsProps {
  isAtBottom: boolean
  isOverflowing: boolean
  scrollToBottom: (forced?: boolean) => void
}

export const ChatScrollButtons: FC<ChatScrollButtonsProps> = ({
  isAtBottom,
  isOverflowing,
  scrollToBottom
}) => {
  return (
    <>
      {!isAtBottom && isOverflowing && (
        <div
          className="border-secondary-foreground bg-secondary cursor-pointer rounded-full border-2 p-1 opacity-75 hover:opacity-100"
          onClick={() => scrollToBottom(true)}
        >
          <IconArrowDown size={18} />
        </div>
      )}
    </>
  )
}
