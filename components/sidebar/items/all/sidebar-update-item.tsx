import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Sheet,
  SheetContent,
  SheetFooter,
  SheetHeader,
  SheetTitle,
  SheetTrigger
} from "@/components/ui/sheet"
import { AssignWorkspaces } from "@/components/workspace/assign-workspaces"
import { ChatbotUIContext } from "@/context/context"
import { getAssistantCollectionsByAssistantId } from "@/db/assistant-collections"
import { getAssistantFilesByAssistantId } from "@/db/assistant-files"
import { getAssistantToolsByAssistantId } from "@/db/assistant-tools"
import { updateChat } from "@/db/chats"
import { getCollectionFilesByCollectionId } from "@/db/collection-files"
import { Tables } from "@/supabase/types"
import { CollectionFile, ContentType, DataItemType } from "@/types"
import { FC, useContext, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { SidebarDeleteItem } from "./sidebar-delete-item"

interface SidebarUpdateItemProps {
  isTyping: boolean
  item: DataItemType
  contentType: ContentType
  children: React.ReactNode
  renderInputs: (renderState: any) => JSX.Element
  updateState: any
}

export const SidebarUpdateItem: FC<SidebarUpdateItemProps> = ({
  item,
  contentType,
  children,
  renderInputs,
  updateState,
  isTyping
}) => {
  const {
    workspaces,
    selectedWorkspace,
    setChats,
    setPresets,
    setPrompts,
    setFiles,
    setCollections,
    setAssistants,
    setTools,
    setModels,
    setAssistantImages
  } = useContext(ChatbotUIContext)

  const buttonRef = useRef<HTMLButtonElement>(null)

  const [isOpen, setIsOpen] = useState(false)
  const [startingWorkspaces, setStartingWorkspaces] = useState<
    Tables<"workspaces">[]
  >([])
  const [selectedWorkspaces, setSelectedWorkspaces] = useState<
    Tables<"workspaces">[]
  >([])

  // Collections Render State
  const [startingCollectionFiles, setStartingCollectionFiles] = useState<
    CollectionFile[]
  >([])
  const [selectedCollectionFiles, setSelectedCollectionFiles] = useState<
    CollectionFile[]
  >([])

  // Assistants Render State
  const [startingAssistantFiles, setStartingAssistantFiles] = useState<
    Tables<"files">[]
  >([])
  const [startingAssistantCollections, setStartingAssistantCollections] =
    useState<Tables<"collections">[]>([])
  const [startingAssistantTools, setStartingAssistantTools] = useState<
    Tables<"tools">[]
  >([])
  const [selectedAssistantFiles, setSelectedAssistantFiles] = useState<
    Tables<"files">[]
  >([])
  const [selectedAssistantCollections, setSelectedAssistantCollections] =
    useState<Tables<"collections">[]>([])
  const [selectedAssistantTools, setSelectedAssistantTools] = useState<
    Tables<"tools">[]
  >([])

  useEffect(() => {
    if (isOpen) {
      const fetchData = async () => {
        const fetchDataFunction = fetchDataFunctions[contentType]
        if (!fetchDataFunction) return
      }

      fetchData()
    }
  }, [isOpen])

  const renderState = {
    chats: null,
    presets: null,
    prompts: null,
    files: null,
    collections: {
      startingCollectionFiles,
      setStartingCollectionFiles,
      selectedCollectionFiles,
      setSelectedCollectionFiles
    },
    assistants: {
      startingAssistantFiles,
      setStartingAssistantFiles,
      startingAssistantCollections,
      setStartingAssistantCollections,
      startingAssistantTools,
      setStartingAssistantTools,
      selectedAssistantFiles,
      setSelectedAssistantFiles,
      selectedAssistantCollections,
      setSelectedAssistantCollections,
      selectedAssistantTools,
      setSelectedAssistantTools
    },
    tools: null,
    models: null
  }

  const fetchDataFunctions = {
    chats: null,
    presets: null,
    prompts: null,
    files: null,
    collections: async (collectionId: string) => {
      const collectionFiles =
        await getCollectionFilesByCollectionId(collectionId)
      setStartingCollectionFiles(collectionFiles.files)
      setSelectedCollectionFiles([])
    },
    assistants: async (assistantId: string) => {
      const assistantFiles = await getAssistantFilesByAssistantId(assistantId)
      setStartingAssistantFiles(assistantFiles.files)

      const assistantCollections =
        await getAssistantCollectionsByAssistantId(assistantId)
      setStartingAssistantCollections(assistantCollections.collections)

      const assistantTools = await getAssistantToolsByAssistantId(assistantId)
      setStartingAssistantTools(assistantTools.tools)

      setSelectedAssistantFiles([])
      setSelectedAssistantCollections([])
      setSelectedAssistantTools([])
    },
    tools: null,
    models: null
  }

  const handleWorkspaceUpdates = async (
    startingWorkspaces: Tables<"workspaces">[],
    selectedWorkspaces: Tables<"workspaces">[],
    itemId: string,
    deleteWorkspaceFn: (
      itemId: string,
      workspaceId: string
    ) => Promise<boolean>,
    createWorkspaceFn: (
      workspaces: { user_id: string; item_id: string; workspace_id: string }[]
    ) => Promise<void>,
    itemIdKey: string
  ) => {
    if (!selectedWorkspace) return

    const deleteList = startingWorkspaces.filter(
      startingWorkspace =>
        !selectedWorkspaces.some(
          selectedWorkspace => selectedWorkspace.id === startingWorkspace.id
        )
    )

    for (const workspace of deleteList) {
      await deleteWorkspaceFn(itemId, workspace.id)
    }

    if (deleteList.map(w => w.id).includes(selectedWorkspace.id)) {
      const setStateFunction = stateUpdateFunctions[contentType]

      if (setStateFunction) {
        setStateFunction((prevItems: any) =>
          prevItems.filter((prevItem: any) => prevItem.id !== item.id)
        )
      }
    }

    const createList = selectedWorkspaces.filter(
      selectedWorkspace =>
        !startingWorkspaces.some(
          startingWorkspace => startingWorkspace.id === selectedWorkspace.id
        )
    )

    await createWorkspaceFn(
      createList.map(workspace => {
        return {
          user_id: workspace.user_id,
          [itemIdKey]: itemId,
          workspace_id: workspace.id
        } as any
      })
    )
  }

  const updateFunctions = {
    chats: updateChat
  }

  const stateUpdateFunctions = {
    chats: setChats,
    presets: setPresets,
    prompts: setPrompts,
    files: setFiles,
    collections: setCollections,
    assistants: setAssistants,
    tools: setTools,
    models: setModels
  }

  const handleUpdate = async () => {
    try {
      const updateFunction = updateFunctions[contentType]
      const setStateFunction = stateUpdateFunctions[contentType]

      if (!updateFunction || !setStateFunction) return
      if (isTyping) return // Prevent update while typing

      const updatedItem = await updateFunction(item.id, updateState)

      setStateFunction((prevItems: any) =>
        prevItems.map((prevItem: any) =>
          prevItem.id === item.id ? updatedItem : prevItem
        )
      )

      setIsOpen(false)

      toast.success(`${contentType.slice(0, -1)} updated successfully`)
    } catch (error) {
      toast.error(`Error updating ${contentType.slice(0, -1)}. ${error}`)
    }
  }

  const handleSelectWorkspace = (workspace: Tables<"workspaces">) => {
    setSelectedWorkspaces(prevState => {
      const isWorkspaceAlreadySelected = prevState.find(
        selectedWorkspace => selectedWorkspace.id === workspace.id
      )

      if (isWorkspaceAlreadySelected) {
        return prevState.filter(
          selectedWorkspace => selectedWorkspace.id !== workspace.id
        )
      } else {
        return [...prevState, workspace]
      }
    })
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (!isTyping && e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      buttonRef.current?.click()
    }
  }

  return (
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>{children}</SheetTrigger>

      <SheetContent
        className="flex min-w-[450px] flex-col justify-between"
        side="left"
        onKeyDown={handleKeyDown}
      >
        <div className="grow overflow-auto">
          <SheetHeader>
            <SheetTitle className="text-2xl font-bold">
              Edit {contentType.slice(0, -1)}
            </SheetTitle>
          </SheetHeader>

          <div className="mt-4 space-y-3">
            {workspaces.length > 1 && (
              <div className="space-y-1">
                <Label>Assigned Workspaces</Label>

                <AssignWorkspaces
                  selectedWorkspaces={selectedWorkspaces}
                  onSelectWorkspace={handleSelectWorkspace}
                />
              </div>
            )}

            {renderInputs(renderState[contentType])}
          </div>
        </div>

        <SheetFooter className="mt-2 flex justify-between">
          <SidebarDeleteItem item={item} contentType={contentType} />

          <div className="flex grow justify-end space-x-2">
            <Button variant="outline" onClick={() => setIsOpen(false)}>
              Cancel
            </Button>

            <Button ref={buttonRef} onClick={handleUpdate}>
              Save
            </Button>
          </div>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
