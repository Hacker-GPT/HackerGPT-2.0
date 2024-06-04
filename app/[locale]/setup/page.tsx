"use client"

import { ChatbotUIContext } from "@/context/context"
import { getProfileByUserId, updateProfile } from "@/db/profile"
import { getHomeWorkspaceByUserId } from "@/db/workspaces"
import { supabase } from "@/lib/supabase/browser-client"
import { TablesUpdate } from "@/supabase/types"
import { useRouter } from "next/navigation"
import { useContext, useEffect } from "react"

export default function SetupPage() {
  const { setProfile } = useContext(ChatbotUIContext)
  const router = useRouter()

  useEffect(() => {
    ;(async () => {
      const session = (await supabase.auth.getSession()).data.session

      if (!session) {
        return router.push("/login")
      } else {
        const user = session.user

        const profile = await getProfileByUserId(user.id)
        setProfile(profile)

        if (!profile.has_onboarded) {
          // Mark the profile as onboarded and save it
          const updateProfilePayload: TablesUpdate<"profiles"> = {
            ...profile,
            has_onboarded: true
          }
          await updateProfile(profile.id, updateProfilePayload)
        }

        // Fetch workspaces and redirect to the home workspace
        const homeWorkspaceId = await getHomeWorkspaceByUserId(user.id)
        return router.push(`/${homeWorkspaceId}/chat`)
      }
    })()
  }, [router, setProfile])

  return null
}
