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
          className="bg-secondary cursor-pointer rounded-full p-1.5 opacity-75 hover:opacity-100"
          onClick={() => scrollToBottom(true)}
        >
          <IconArrowDown size={18} />
        </div>
      )}
    </>
  )
}
