import { ChatbotUIContext } from "@/context/context"
import { Tables } from "@/supabase/types"
import { useContext } from "react"

export const usePromptAndCommand = () => {
  const {
    chatFiles,
    setNewMessageFiles,
    userInput,
    setUserInput,
    setShowFilesDisplay,
    setIsAtPickerOpen,
    setSlashCommand,
    setAtCommand,
    setUseRetrieval,
    setToolCommand,
    setIsToolPickerOpen,
    setSelectedTools
  } = useContext(ChatbotUIContext)

  const handleInputChange = (value: string) => {
    const slashTextRegex = /\/([^ ]*)$/
    const atTextRegex = /#([^ ]*)$/
    const toolTextRegex = /!([^ ]*)$/
    const slashMatch = value.match(slashTextRegex)
    const atMatch = value.match(atTextRegex)
    const toolMatch = value.match(toolTextRegex)

    if (slashMatch) {
      setSlashCommand(slashMatch[1])
    } else if (atMatch) {
      setIsAtPickerOpen(true)
      setAtCommand(atMatch[1])
    } else if (toolMatch) {
      setIsToolPickerOpen(true)
      setToolCommand(toolMatch[1])
    } else {
      setIsAtPickerOpen(false)
      setSlashCommand("")
      setAtCommand("")
      setIsToolPickerOpen(false)
      setToolCommand("")
    }

    setUserInput(value)
  }

  const handleSelectUserFile = async (file: Tables<"files">) => {
    setShowFilesDisplay(true)
    setIsAtPickerOpen(false)
    setUseRetrieval(true)

    setNewMessageFiles(prev => {
      const fileAlreadySelected =
        prev.some(prevFile => prevFile.id === file.id) ||
        chatFiles.some(chatFile => chatFile.id === file.id)

      if (!fileAlreadySelected) {
        return [
          ...prev,
          {
            id: file.id,
            name: file.name,
            type: file.type,
            file: null
          }
        ]
      }
      return prev
    })

    setUserInput(userInput.replace(/#[^ ]*$/, ""))
  }

  const handleSelectTool = (tool: Tables<"tools">) => {
    setIsToolPickerOpen(false)
    setUserInput(userInput.replace(/![^ ]*$/, ""))
    setSelectedTools(prev => [...prev, tool])
  }

  return {
    handleInputChange,
    handleSelectUserFile,
    handleSelectTool
  }
}
