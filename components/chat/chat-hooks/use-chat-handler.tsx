import { useAlertContext } from "@/context/alert-context"
import { ChatbotUIContext } from "@/context/context"
import { updateChat } from "@/db/chats"
import { getFileById } from "@/db/files"
import { Tables, TablesInsert } from "@/supabase/types"
import { ChatMessage, ChatPayload, LLMID, ModelProvider } from "@/types"
import { PluginID } from "@/types/plugins"
import { useRouter } from "next/navigation"
import { useContext, useEffect, useRef } from "react"
import { toast } from "sonner"
import { LLM_LIST } from "../../../lib/models/llm/llm-list"

import { createMessageFeedback } from "@/db/message-feedback"
import {
  createTempMessages,
  handleCreateChat,
  handleCreateMessages,
  handleHostedChat,
  handleHostedPluginsChat,
  handleRetrieval,
  validateChatSettings
} from "../chat-helpers"
import { usePromptAndCommand } from "./use-prompt-and-command"

export const useChatHandler = () => {
  const router = useRouter()
  const { dispatch: alertDispatch } = useAlertContext()
  const { handleSelectUserFile } = usePromptAndCommand()

  const {
    userInput,
    chatFiles,
    setUserInput,
    setNewMessageImages,
    profile,
    setIsGenerating,
    setChatMessages,
    setFirstTokenReceived,
    selectedChat,
    selectedWorkspace,
    setSelectedChat,
    setChats,
    abortController,
    setAbortController,
    chatSettings,
    newMessageImages,
    chatMessages,
    chatImages,
    setChatImages,
    setChatFiles,
    setNewMessageFiles,
    setShowFilesDisplay,
    newMessageFiles,
    setToolInUse,
    setFiles,
    useRetrieval,
    sourceCount,
    setIsAtPickerOpen,
    setChatSettings,
    isAtPickerOpen,
    subscription,
    isRagEnabled,
    isGenerating,
    setUseRetrieval,
    setIsReadyToChat
  } = useContext(ChatbotUIContext)

  let { selectedPlugin } = useContext(ChatbotUIContext)

  const isGeneratingRef = useRef(isGenerating)

  useEffect(() => {
    isGeneratingRef.current = isGenerating
  }, [isGenerating])

  const chatInputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!isAtPickerOpen) {
      chatInputRef.current?.focus()
    }
  }, [isAtPickerOpen])

  // Initialize chat settings on component mount
  useEffect(() => {
    if (selectedChat && selectedChat.model) {
      setChatSettings(prevSettings => ({
        ...prevSettings,
        model: selectedChat.model as LLMID
      }))
    }
  }, [selectedChat, setChatSettings])

  const handleSelectChat = async (chat: Tables<"chats">) => {
    if (!selectedWorkspace) return
    await handleStopMessage()
    setIsReadyToChat(false)

    if (chat.model) {
      setChatSettings(prevSettings => ({
        ...prevSettings,
        model: chat.model as LLMID
      }))
    }

    return router.push(`/${selectedWorkspace.id}/chat/${chat.id}`)
  }

  const handleNewChat = async () => {
    if (!selectedWorkspace) return

    await handleStopMessage()

    setUserInput("")
    setChatMessages([])
    setSelectedChat(null)

    setIsGenerating(false)
    setFirstTokenReceived(false)

    setChatFiles([])
    setChatImages([])
    setNewMessageFiles([])
    setNewMessageImages([])
    setShowFilesDisplay(false)
    setIsAtPickerOpen(false)
    setUseRetrieval(false)

    setToolInUse("none")

    setIsReadyToChat(true)
    return router.push(`/${selectedWorkspace.id}/chat`)
  }

  const handleFocusChatInput = () => {
    chatInputRef.current?.focus()
  }

  const handleStopMessage = async () => {
    if (abortController && !abortController.signal.aborted) {
      abortController.abort()
      while (isGeneratingRef.current) {
        await new Promise(resolve => setTimeout(resolve, 100))
      }
      await new Promise(resolve => setTimeout(resolve, 100))
    }
  }

  const handleSendFeedback = async (
    chatMessage: ChatMessage,
    feedback: "good" | "bad",
    reason?: string,
    detailedFeed?: string,
    allow_email?: boolean,
    allow_sharing?: boolean
  ) => {
    const feedbackInsert: TablesInsert<"feedback"> = {
      message_id: chatMessage.message.id,
      user_id: chatMessage.message.user_id,
      chat_id: chatMessage.message.chat_id,
      feedback: feedback,
      reason: reason ?? chatMessage.feedback?.reason,
      detailed_feedback:
        detailedFeed ?? chatMessage.feedback?.detailed_feedback,
      model: chatMessage.message.model,
      created_at: chatMessage.feedback?.created_at || new Date().toISOString(),
      updated_at: new Date().toISOString(),
      sequence_number: chatMessage.message.sequence_number,
      allow_email: allow_email,
      allow_sharing: allow_sharing,
      has_files: chatMessage.fileItems.length > 0,
      plugin: chatMessage.message.plugin || PluginID.NONE,
      rag_used: chatMessage.message.rag_used,
      rag_id: chatMessage.message.rag_id
    }
    const newFeedback = await createMessageFeedback(feedbackInsert)
    setChatMessages((prevMessages: ChatMessage[]) =>
      prevMessages.map((message: ChatMessage) =>
        message.message.id === chatMessage.message.id
          ? { ...message, feedback: newFeedback[0] }
          : message
      )
    )
  }

  const handleSendContinuation = async () => {
    await handleSendMessage(null, chatMessages, false, true)
  }

  const handleSendMessage = async (
    messageContent: string | null,
    chatMessages: ChatMessage[],
    isRegeneration: boolean,
    isContinuation: boolean = false,
    editSequenceNumber?: number,
    model?: LLMID
  ) => {
    const isEdit = editSequenceNumber !== undefined

    try {
      if (!isRegeneration) {
        setUserInput("")
      }
      setIsGenerating(true)
      setIsAtPickerOpen(false)
      setNewMessageImages([])

      const newAbortController = new AbortController()
      setAbortController(newAbortController)

      const modelData = [...LLM_LIST].find(
        llm => llm.modelId === (model || chatSettings?.model)
      )

      validateChatSettings(
        chatSettings,
        modelData,
        profile,
        selectedWorkspace,
        isContinuation,
        messageContent
      )

      if (chatSettings && !isRegeneration) {
        chatSettings.model = model || chatSettings.model
      }

      if (!isContinuation && messageContent && subscription) {
        const urlRegex = /https?:\/\/[^\s]+/g
        const urls = messageContent.match(urlRegex) || []

        if (selectedPlugin === PluginID.WEB_SCRAPER && urls.length > 0) {
          setToolInUse(PluginID.WEB_SCRAPER)
          for (const url of urls) {
            try {
              const response = await fetch(`/api/retrieval/process/web`, {
                method: "POST",
                body: JSON.stringify({
                  embeddingsProvider: "openai",
                  workspace_id: selectedWorkspace?.id,
                  url: url
                })
              })
              const { message, fileId } = await response.json()

              if (message === "Embed Successful") {
                const fileFromDb = await getFileById(fileId)
                if (
                  fileFromDb &&
                  !newMessageFiles.some(file => file.id === fileFromDb.id) &&
                  !chatFiles.some(file => file.id === fileFromDb.id)
                ) {
                  handleSelectUserFile(fileFromDb)
                  setFiles(prevFiles => [...prevFiles, fileFromDb])
                  newMessageFiles.push({ ...fileFromDb, file: null })
                } else {
                  toast.error("File not found in database.")
                }
              } else {
                toast.error("Failed to process websites.")
              }
            } catch (error) {
              toast.error("An error occurred while processing the URL.")
            }
          }
        }
      }

      let currentChat = selectedChat ? { ...selectedChat } : null

      const b64Images = newMessageImages.map(image => image.base64)

      const { tempUserChatMessage, tempAssistantChatMessage } =
        createTempMessages(
          messageContent,
          chatMessages,
          chatSettings!,
          b64Images,
          isContinuation,
          selectedPlugin,
          model || chatSettings!.model
        )

      let sentChatMessages = [...chatMessages]

      // If the message is an edit, remove all following messages
      if (isEdit) {
        sentChatMessages = sentChatMessages.filter(
          chatMessage =>
            chatMessage.message.sequence_number < editSequenceNumber
        )
      }

      if (isRegeneration) {
        sentChatMessages.pop()
        sentChatMessages.push(tempAssistantChatMessage)
      } else {
        sentChatMessages.push(tempUserChatMessage)
        if (!isContinuation) sentChatMessages.push(tempAssistantChatMessage)
      }

      // Update the UI with the new messages
      setChatMessages(sentChatMessages)

      let retrievedFileItems: Tables<"file_items">[] = []

      if (
        (newMessageFiles.length > 0 || chatFiles.length > 0) &&
        (useRetrieval || selectedPlugin === PluginID.WEB_SCRAPER) &&
        !isContinuation
      ) {
        if (selectedPlugin !== PluginID.WEB_SCRAPER) {
          setToolInUse("retrieval")
        }

        retrievedFileItems = await handleRetrieval(
          userInput,
          newMessageFiles,
          chatFiles,
          chatSettings!.embeddingsProvider,
          sourceCount
        )
      }

      let payload: ChatPayload = {
        chatSettings: {
          ...chatSettings!,
          model: model || chatSettings!.model
        },
        chatMessages: sentChatMessages,
        messageFileItems: retrievedFileItems
      }

      let generatedText = ""
      let finishReasonFromResponse = ""
      let ragUsed = false
      let ragId = null

      // if (!isContinuation && selectedPlugin === PluginID.AUTO_PLUGIN_SELECTOR) {
      //   selectedPlugin = await handleDetectPlugin(payload, selectedPlugin, true)
      // } else if (!isContinuation && selectedPlugin === PluginID.NONE) {
      if (!isContinuation && selectedPlugin === PluginID.NONE) {
        selectedPlugin = await handleDetectPlugin(
          payload,
          selectedPlugin,
          false
        )
      }

      if (
        selectedPlugin.length > 0 &&
        selectedPlugin !== PluginID.NONE &&
        // selectedPlugin !== PluginID.AUTO_PLUGIN_SELECTOR &&
        selectedPlugin !== PluginID.CODE_LLM &&
        selectedPlugin !== PluginID.WEB_SEARCH &&
        selectedPlugin !== PluginID.WEB_SCRAPER
      ) {
        let fileData: { fileName: string; fileContent: string }[] = []

        const nonExcludedPluginsForFilesCommand = [
          PluginID.NUCLEI,
          PluginID.ALTERX,
          PluginID.DNSX,
          PluginID.HTTPX,
          PluginID.KATANA
        ]

        const isCommand = (allowedCommands: string[], message: string) => {
          if (!message.startsWith("/")) return false
          const trimmedMessage = message.trim().toLowerCase()

          // Check if the message matches any of the allowed commands
          return allowedCommands.some(commandName => {
            const commandPattern = new RegExp(
              `^\\/${commandName}(?:\\s+(-[a-z]+|\\S+))*$`
            )
            return commandPattern.test(trimmedMessage)
          })
        }

        if (
          messageContent &&
          newMessageFiles.length > 0 &&
          newMessageFiles[0].type === "text" &&
          (nonExcludedPluginsForFilesCommand.includes(selectedPlugin) ||
            isCommand(nonExcludedPluginsForFilesCommand, messageContent))
        ) {
          const fileIds = newMessageFiles
            .filter(file => file.type === "text")
            .map(file => file.id)

          if (fileIds.length > 0) {
            const response = await fetch(`/api/retrieval/file-2v`, {
              method: "POST",
              headers: {
                "Content-Type": "application/json"
              },
              body: JSON.stringify({ fileIds: fileIds })
            })

            if (!response.ok) {
              const errorData = await response.json()
              toast.warning(errorData.message)
            }

            const data = await response.json()
            fileData.push(...data.files)
          }
        }

        const { fullText, finishReason } = await handleHostedPluginsChat(
          payload,
          profile!,
          modelData!,
          tempAssistantChatMessage,
          isRegeneration,
          newAbortController,
          newMessageImages,
          chatImages,
          setIsGenerating,
          setFirstTokenReceived,
          setChatMessages,
          setToolInUse,
          alertDispatch,
          selectedPlugin,
          fileData
        )
        generatedText = fullText
        finishReasonFromResponse = finishReason
      } else {
        const {
          fullText,
          finishReason,
          ragUsed: ragUsedFromResponse,
          ragId: ragIdFromResponse
        } = await handleHostedChat(
          payload,
          modelData!,
          tempAssistantChatMessage,
          isRegeneration,
          isRagEnabled,
          isContinuation,
          newAbortController,
          chatImages,
          setIsGenerating,
          setFirstTokenReceived,
          setChatMessages,
          setToolInUse,
          alertDispatch,
          selectedPlugin
        )
        generatedText = fullText
        finishReasonFromResponse = finishReason
        ragUsed = ragUsedFromResponse
        ragId = ragIdFromResponse
      }

      if (!currentChat) {
        currentChat = await handleCreateChat(
          chatSettings!,
          profile!,
          selectedWorkspace!,
          messageContent || "",
          newMessageFiles,
          finishReasonFromResponse,
          setSelectedChat,
          setChats,
          setChatFiles
        )
      } else {
        const updatedChat = await updateChat(currentChat.id, {
          updated_at: new Date().toISOString(),
          finish_reason: finishReasonFromResponse,
          model: chatSettings?.model
        })

        setChats(prevChats => {
          const updatedChats = prevChats.map(prevChat =>
            prevChat.id === updatedChat.id ? updatedChat : prevChat
          )

          return updatedChats
        })

        if (selectedChat?.id === updatedChat.id) {
          setSelectedChat(updatedChat)
        }
      }

      await handleCreateMessages(
        chatMessages,
        currentChat,
        profile!,
        modelData!,
        messageContent,
        generatedText,
        newMessageImages,
        isRegeneration,
        isContinuation,
        retrievedFileItems,
        setChatMessages,
        setChatImages,
        selectedPlugin,
        editSequenceNumber,
        ragUsed,
        ragId
      )

      setIsGenerating(false)
      setFirstTokenReceived(false)
    } catch (error) {
      setIsGenerating(false)
      setFirstTokenReceived(false)
      // Restore the chat messages to the previous state
      setChatMessages(chatMessages)
    }
  }

  const handleSendEdit = async (
    editedContent: string,
    sequenceNumber: number
  ) => {
    if (!selectedChat) return

    handleSendMessage(editedContent, chatMessages, false, false, sequenceNumber)
  }

  const handleDetectPlugin = async (
    payload: ChatPayload,
    selectedPlugin: PluginID,
    useAllPlugins: boolean
  ) => {
    const response = await fetch("/api/v2/chat/plugin-detector", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ payload, selectedPlugin, useAllPlugins })
    })

    if (response.ok) {
      const { plugin: detectedPlugin } = await response.json()
      if (detectedPlugin && detectedPlugin !== "None") {
        return detectedPlugin
      }
    }
    return selectedPlugin
  }

  return {
    chatInputRef,
    handleNewChat,
    handleSendMessage,
    handleFocusChatInput,
    handleStopMessage,
    handleSendContinuation,
    handleSendEdit,
    handleSendFeedback,
    handleSelectChat
  }
}
