import { FC } from "react"
import { Label } from "@/components/ui/label"
import { TextareaAutosize } from "@/components/ui/textarea-autosize"
import { LimitDisplay } from "@/components/ui/limit-display"
import { PROFILE_CONTEXT_MAX } from "@/db/limits"
import { TabsContent } from "@/components/ui/tabs"

interface PersonalizationTabProps {
  value: string
  profileInstructions: string
  setProfileInstructions: (value: string) => void
}

export const PersonalizationTab: FC<PersonalizationTabProps> = ({
  value,
  profileInstructions,
  setProfileInstructions
}) => {
  const isOverLimit = profileInstructions.length > PROFILE_CONTEXT_MAX

  return (
    <TabsContent className="mt-4 space-y-4" value={value}>
      <div className="space-y-1">
        <Label className="text-sm">
          What would you like HackerGPT to know about you to provide better
          responses?
        </Label>

        <TextareaAutosize
          value={profileInstructions}
          onValueChange={setProfileInstructions}
          placeholder="Profile context..."
          minRows={6}
          maxRows={10}
          className={isOverLimit ? "border-red-500" : ""}
        />

        <LimitDisplay
          used={profileInstructions.length}
          limit={PROFILE_CONTEXT_MAX}
          isOverLimit={isOverLimit}
        />
      </div>
    </TabsContent>
  )
}
