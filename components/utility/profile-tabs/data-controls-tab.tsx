import { FC } from "react"
import { Button } from "@/components/ui/button"
import { TabsContent } from "@/components/ui/tabs"

interface DataControlsTabProps {
  value: string
  onDeleteAccount: () => void
}

export const DataControlsTab: FC<DataControlsTabProps> = ({
  value,
  onDeleteAccount
}) => {
  return (
    <TabsContent className="space-y-4" value={value}>
      <div className="space-y-4">
        <h2 className="text-lg font-semibold">Account Deletion</h2>
        <p className="text-muted-foreground text-sm">
          Warning: This action is irreversible. All your data will be
          permanently deleted.
        </p>
        <Button variant="destructive" onClick={onDeleteAccount}>
          Delete Account
        </Button>
      </div>
    </TabsContent>
  )
}
