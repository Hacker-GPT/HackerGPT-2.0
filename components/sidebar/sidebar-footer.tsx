import React, { FC, useContext, useState } from "react"
import { ChatbotUIContext } from "@/context/context"
import { PlanDialog } from "../utility/plan-dialog"
import { WithTooltip } from "../ui/with-tooltip"
import { ProfileSettings } from "../utility/profile-settings"
import { IconLockOpen } from "@tabler/icons-react"
import { SIDEBAR_ICON_SIZE } from "../sidebar/sidebar-switcher"

export const SidebarFooter: FC = () => {
  const { subscription } = useContext(ChatbotUIContext)
  const [showPlanDialog, setShowPlanDialog] = useState(false)

  return (
    <div className="mb-1 ml-2 mt-3 flex flex-col items-start justify-between">
      {!subscription && (
        <div
          className="mb-4 flex cursor-pointer items-center hover:opacity-50"
          onClick={() => setShowPlanDialog(true)}
        >
          <IconLockOpen size={SIDEBAR_ICON_SIZE} />
          <span className="text-md ml-2 font-medium">Upgrade to Pro</span>
        </div>
      )}
      <WithTooltip
        display={<div>Profile Settings</div>}
        trigger={<ProfileSettings />}
      />
      <PlanDialog
        showIcon={false}
        open={showPlanDialog}
        onOpenChange={setShowPlanDialog}
      />
    </div>
  )
}
