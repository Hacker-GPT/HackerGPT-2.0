import { FC, useRef, useEffect } from "react"
import { Button } from "../../ui/button"
import { Input } from "../../ui/input"
import { Label } from "../../ui/label"
import { ThemeSwitcher } from "../theme-switcher"
import { IconLogout } from "@tabler/icons-react"

interface ProfileTabProps {
  userEmail: string
  handleDeleteAllChats: () => void
  handleSignOut: () => void
  isMobile: boolean
}

export const ProfileTab: FC<ProfileTabProps> = ({
  userEmail,
  handleDeleteAllChats,
  handleSignOut,
  isMobile
}) => {
  const inputRef = useRef<HTMLInputElement>(null)
  const isLongEmail = userEmail.length > 30

  useEffect(() => {
    if (inputRef.current) {
      const width = Math.max(userEmail.length - 4, 10)
      inputRef.current.style.width = `${width}ch`
    }
  }, [userEmail])

  return (
    <div className="space-y-4">
      <div
        className={
          isLongEmail && isMobile
            ? "space-y-2"
            : "flex items-center justify-between"
        }
      >
        <Label htmlFor="email-input">Email address</Label>
        <Input
          ref={inputRef}
          id="email-input"
          value={userEmail}
          readOnly
          className="cursor-default"
        />
      </div>

      <div className="flex items-center justify-between">
        <Label>Theme</Label>
        <ThemeSwitcher />
      </div>

      <div className="flex items-center justify-between">
        <Label>Delete all chats</Label>
        <Button variant="destructive" onClick={handleDeleteAllChats}>
          Delete all
        </Button>
      </div>

      <div className="flex items-center justify-between">
        <Label>Log out</Label>
        <Button
          variant="outline"
          onClick={handleSignOut}
          className="flex items-center"
        >
          <IconLogout className="mr-2" size={18} />
          Log out
        </Button>
      </div>
    </div>
  )
}
