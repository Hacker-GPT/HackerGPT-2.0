import { FC } from "react"
import { DialogPanel, DialogTitle } from "@headlessui/react"
import { Button } from "../ui/button"
import { TransitionedDialog } from "../ui/transitioned-dialog"

interface DeleteAllChatsDialogProps {
  isOpen: boolean
  onClose: () => void
  onConfirm: () => void
}

export const DeleteAllChatsDialog: FC<DeleteAllChatsDialogProps> = ({
  isOpen,
  onClose,
  onConfirm
}) => {
  return (
    <TransitionedDialog isOpen={isOpen} onClose={onClose}>
      <DialogPanel className="bg-popover w-full max-w-md overflow-hidden rounded-2xl p-6 text-left align-middle shadow-xl transition-all">
        <DialogTitle
          as="h3"
          className="text-center text-lg font-medium leading-6"
        >
          Delete All Chats
        </DialogTitle>
        <div className="mt-2">
          <p className="text-center text-sm">
            Are you sure you want to delete all chats? This action cannot be
            undone.
          </p>
        </div>

        <div className="mt-4 flex justify-center space-x-4">
          <Button onClick={onClose}>Cancel</Button>
          <Button variant="destructive" onClick={onConfirm}>
            Delete All
          </Button>
        </div>
      </DialogPanel>
    </TransitionedDialog>
  )
}
