import { FC, Fragment } from "react"
import {
  Dialog,
  Transition,
  DialogPanel,
  DialogTitle,
  TransitionChild
} from "@headlessui/react"
import { Button } from "../ui/button"

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
    <Transition show={isOpen} as={Fragment}>
      <Dialog onClose={onClose} className="relative z-50">
        <TransitionChild
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-50 backdrop-blur-sm dark:bg-opacity-75" />
        </TransitionChild>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4 text-center">
            <TransitionChild
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <DialogPanel className="bg-primary-foreground w-full max-w-md overflow-hidden rounded-2xl p-6 text-left align-middle shadow-xl transition-all">
                <DialogTitle
                  as="h3"
                  className="text-center text-lg font-medium leading-6"
                >
                  Delete All Chats
                </DialogTitle>
                <div className="mt-2">
                  <p className="text-center text-sm">
                    Are you sure you want to delete all chats? This action
                    cannot be undone.
                  </p>
                </div>

                <div className="mt-4 flex justify-center space-x-4">
                  <Button onClick={onClose}>Cancel</Button>
                  <Button variant="destructive" onClick={onConfirm}>
                    Delete All
                  </Button>
                </div>
              </DialogPanel>
            </TransitionChild>
          </div>
        </div>
      </Dialog>
    </Transition>
  )
}
