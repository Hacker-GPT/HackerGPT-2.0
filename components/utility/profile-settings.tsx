import { ChatbotUIContext } from "@/context/context"
import { PROFILE_DISPLAY_NAME_MAX, PROFILE_CONTEXT_MAX } from "@/db/limits"
import { updateProfile } from "@/db/profile"
import { LLM_LIST_MAP } from "@/lib/models/llm/llm-list"
import { supabase } from "@/lib/supabase/browser-client"
import { IconLogout, IconSettings, IconTrash } from "@tabler/icons-react"
import Image from "next/image"
import { useRouter } from "next/navigation"
import { FC, useCallback, useContext, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { SIDEBAR_ICON_SIZE } from "../sidebar/sidebar-switcher"
import { Button } from "../ui/button"
import { Input } from "../ui/input"
import { Label } from "../ui/label"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from "../ui/sheet"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs"
import { ThemeSwitcher } from "./theme-switcher"
import { SubscriptionTab } from "./profile-tabs/subscription-tab"
import Modal from "@/components/chat/dialog-portal"
import { deleteAllChats } from "@/db/chats"
import { PersonalizationTab } from "./profile-tabs/personalization-tab"

interface ProfileSettingsProps {}

export const ProfileSettings: FC<ProfileSettingsProps> = ({}) => {
  const {
    profile,
    setProfile,
    envKeyMap,
    setAvailableHostedModels,
    setAvailableOpenRouterModels,
    isMobile
  } = useContext(ChatbotUIContext)

  const [showConfirmationDialog, setShowConfirmationDialog] =
    useState<boolean>(false)

  const router = useRouter()

  const buttonRef = useRef<HTMLButtonElement>(null)

  const [isOpen, setIsOpen] = useState(false)

  const [displayName, setDisplayName] = useState(profile?.display_name || "")
  const [userEmail, setUserEmail] = useState("")

  const [profileInstructions, setProfileInstructions] = useState(
    profile?.profile_context || ""
  )

  const handleSignOut = async () => {
    await supabase.auth.signOut()
    router.push("/login")
    router.refresh()
    return
  }

  useEffect(() => {
    const fetchUserEmail = async () => {
      const user = await supabase.auth.getUser()
      setUserEmail(user?.data.user?.email || "Not available")
    }
    fetchUserEmail()
  }, [])

  const handleSave = async () => {
    if (!profile) return

    const isOverLimit = profileInstructions.length > PROFILE_CONTEXT_MAX
    if (isOverLimit) {
      toast.error(
        `Profile instructions exceed the limit of ${PROFILE_CONTEXT_MAX} characters.`
      )
      return
    }

    const updatedProfile = await updateProfile(profile.id, {
      ...profile,
      display_name: displayName,
      profile_context: profileInstructions
    })

    setProfile(updatedProfile)

    toast.success("Profile updated!", { duration: 2000 })

    const providers = [
      "openai",
      "google",
      "azure",
      "anthropic",
      "mistral",
      "perplexity",
      "openrouter"
    ]

    providers.forEach(async provider => {
      let providerKey: keyof typeof profile

      if (provider === "google") {
        providerKey = "google_gemini_api_key"
      } else if (provider === "azure") {
        providerKey = "azure_openai_api_key"
      } else {
        providerKey = `${provider}_api_key` as keyof typeof profile
      }

      const models = LLM_LIST_MAP[provider]
      const envKeyActive = envKeyMap[provider]

      if (!envKeyActive) {
        const hasApiKey = !!updatedProfile[providerKey]

        if (provider === "openrouter") {
          // if (hasApiKey && availableOpenRouterModels.length === 0) {
          //   const openrouterModels: OpenRouterLLM[] =
          //     await fetchOpenRouterModels()
          //   setAvailableOpenRouterModels(prev => {
          //     const newModels = openrouterModels.filter(
          //       model =>
          //         !prev.some(prevModel => prevModel.modelId === model.modelId)
          //     )
          //     return [...prev, ...newModels]
          //   })
          // } else {
          setAvailableOpenRouterModels([])
          // }
        } else {
          if (hasApiKey && Array.isArray(models)) {
            setAvailableHostedModels(prev => {
              const newModels = models.filter(
                model =>
                  !prev.some(prevModel => prevModel.modelId === model.modelId)
              )
              return [...prev, ...newModels]
            })
          } else if (!hasApiKey && Array.isArray(models)) {
            setAvailableHostedModels(prev =>
              prev.filter(model => !models.includes(model))
            )
          }
        }
      }
    })

    setIsOpen(false)
  }

  const debounce = (func: (...args: any[]) => void, wait: number) => {
    let timeout: NodeJS.Timeout | null

    return (...args: any[]) => {
      const later = () => {
        if (timeout) clearTimeout(timeout)
        func(...args)
      }

      if (timeout) clearTimeout(timeout)
      timeout = setTimeout(later, wait)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter") {
      buttonRef.current?.click()
    }
  }

  const handleConfirm = async () => {
    setShowConfirmationDialog(false)
    const deleted = await deleteAllChats(profile?.user_id || "")
    if (deleted) {
      window.location.reload()
    } else {
      toast.error("Failed to delete all chats")
    }
  }

  const handleCancel = () => {
    setIsOpen(true)
    setShowConfirmationDialog(false)
  }

  const handleDeleteAllChats = async () => {
    setIsOpen(false)
    showConfirmationDialog
      ? setShowConfirmationDialog(false)
      : setShowConfirmationDialog(true)
  }

  if (!profile) return null

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        {profile.image_url ? (
          <Image
            className="mt-2 size-[34px] cursor-pointer rounded hover:opacity-50"
            src={profile.image_url + "?" + new Date().getTime()}
            height={34}
            width={34}
            alt={"Image"}
          />
        ) : (
          <Button size="icon" variant="ghost">
            <IconSettings size={SIDEBAR_ICON_SIZE} />
          </Button>
        )}
      </SheetTrigger>

      <Modal isOpen={showConfirmationDialog}>
        <div className="size-screen fixed inset-0 z-50 bg-black bg-opacity-50 backdrop-blur-sm dark:bg-opacity-75"></div>

        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div className="bg-background w-full max-w-lg rounded-md p-10 text-center">
            <p>
              <b>All chats and folders</b> will be deleted!
            </p>
            <p>Are you sure you want to do this?</p>
            <div className="mt-5 flex justify-center gap-5">
              <Button onClick={handleCancel} variant="default">
                Cancel
              </Button>
              <Button onClick={handleConfirm} variant="destructive">
                Delete All
              </Button>
            </div>
          </div>
        </div>
      </Modal>

      <SheetContent
        className="flex flex-col justify-between"
        side="left"
        onKeyDown={handleKeyDown}
      >
        <div className="grow overflow-auto">
          <SheetHeader>
            <SheetTitle className="flex items-center justify-between space-x-2">
              <div>Settings</div>

              <Button
                tabIndex={-1}
                className="text-xs"
                size="sm"
                onClick={handleSignOut}
              >
                <IconLogout className="mr-1" size={20} />
                Logout
              </Button>
            </SheetTitle>
          </SheetHeader>

          <Tabs defaultValue="profile">
            <TabsList
              className={`mt-4 grid w-full gap-2 ${isMobile ? "grid-cols-2" : "grid-cols-[auto,auto,auto]"}`}
            >
              <TabsTrigger value="profile">General</TabsTrigger>
              <TabsTrigger value="subscription">Subscription</TabsTrigger>
              {!isMobile && (
                <TabsTrigger value="personalization">
                  Personalization
                </TabsTrigger>
              )}
            </TabsList>

            {isMobile && (
              <TabsList className="grid w-full grid-cols-1 gap-2">
                <TabsTrigger value="personalization">
                  Personalization
                </TabsTrigger>
              </TabsList>
            )}

            <TabsContent className="mt-4 space-y-4" value="profile">
              <div className="space-y-1">
                <Label>Email</Label>
                <Input
                  value={userEmail}
                  disabled={true}
                  className="cursor-not-allowed"
                />
              </div>

              <div className="space-y-1">
                <Label>Chat Display Name</Label>

                <Input
                  placeholder="Chat display name..."
                  value={displayName}
                  onChange={e => setDisplayName(e.target.value)}
                  maxLength={PROFILE_DISPLAY_NAME_MAX}
                />
              </div>

              <div className="flex items-center justify-between">
                <Label>Delete All Chats & Folders</Label>
                <Button
                  tabIndex={-1}
                  variant="destructive"
                  onClick={handleDeleteAllChats}
                >
                  Delete All
                </Button>
              </div>
            </TabsContent>

            <SubscriptionTab value="subscription" />

            <PersonalizationTab
              value="personalization"
              profileInstructions={profileInstructions}
              setProfileInstructions={setProfileInstructions}
            />
          </Tabs>
        </div>

        <div className="mt-6 flex items-center">
          <div className="flex items-center space-x-1">
            <ThemeSwitcher />
          </div>

          <div className="ml-auto space-x-2">
            <Button variant="ghost" onClick={() => setIsOpen(false)}>
              Cancel
            </Button>

            <Button ref={buttonRef} onClick={handleSave}>
              Save
            </Button>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
