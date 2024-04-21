import { useAlertContext } from "@/context/alert-context"
import { ChatbotUIContext } from "@/context/context"
import { updateChat } from "@/db/chats"
import { getFileById } from "@/db/files"
import { deleteMessagesIncludingAndAfter } from "@/db/messages"
import { Tables, TablesInsert } from "@/supabase/types"
import { ChatMessage, ChatPayload, LLMID, ModelProvider } from "@/types"
import { PluginID } from "@/types/plugins"
import { useRouter } from "next/navigation"
import { useContext, useEffect, useRef } from "react"
import { toast } from "sonner"
import { LLM_LIST } from "../../../lib/models/llm/llm-list"

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
import { create } from "domain"
import { createMessageFeedback } from "@/db/message-feedback"

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
    setSelectedTools,
    availableLocalModels,
    availableOpenRouterModels,
    abortController,
    setAbortController,
    chatSettings,
    newMessageImages,
    selectedAssistant,
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
    setIsPromptPickerOpen,
    setIsAtPickerOpen,
    selectedTools,
    selectedPreset,
    setChatSettings,
    models,
    isPromptPickerOpen,
    isAtPickerOpen,
    isToolPickerOpen,
    selectedPlugin,
    subscription
  } = useContext(ChatbotUIContext)

  const chatInputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    if (!isPromptPickerOpen || !isAtPickerOpen || !isToolPickerOpen) {
      chatInputRef.current?.focus()
    }
  }, [isPromptPickerOpen, isAtPickerOpen, isToolPickerOpen])

  const handleNewChat = async () => {
    if (!selectedWorkspace) return

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
    setIsPromptPickerOpen(false)
    setIsAtPickerOpen(false)

    setSelectedTools([])
    setToolInUse("none")

    if (selectedAssistant) {
      setChatSettings({
        model: selectedAssistant.model as LLMID,
        prompt: selectedAssistant.prompt,
        temperature: selectedAssistant.temperature,
        contextLength: selectedAssistant.context_length,
        includeProfileContext: selectedAssistant.include_profile_context,
        includeWorkspaceInstructions:
          selectedAssistant.include_workspace_instructions,
        embeddingsProvider: selectedAssistant.embeddings_provider as
          | "openai"
          | "local"
      })
    } else if (selectedPreset) {
      setChatSettings({
        model: selectedPreset.model as LLMID,
        prompt: selectedPreset.prompt,
        temperature: selectedPreset.temperature,
        contextLength: selectedPreset.context_length,
        includeProfileContext: selectedPreset.include_profile_context,
        includeWorkspaceInstructions:
          selectedPreset.include_workspace_instructions,
        embeddingsProvider: selectedPreset.embeddings_provider as
          | "openai"
          | "local"
      })
    } else if (selectedWorkspace) {
      // setChatSettings({
      //   model: "mistral-medium" as LLMID,
      //   prompt:
      //     selectedWorkspace.default_prompt ||
      //     "You are a friendly, helpful AI assistant.",
      //   temperature: selectedWorkspace.default_temperature || 0.4,
      //   contextLength: selectedWorkspace.default_context_length || 4096,
      //   includeProfileContext:
      //     selectedWorkspace.include_profile_context || true,
      //   includeWorkspaceInstructions:
      //     selectedWorkspace.include_workspace_instructions || true,
      //   embeddingsProvider:
      //     (selectedWorkspace.embeddings_provider as "openai" | "local") ||
      //     "openai"
      // })
    }

    return router.push(`/${selectedWorkspace.id}/chat`)
  }

  const handleFocusChatInput = () => {
    chatInputRef.current?.focus()
  }

  const handleStopMessage = () => {
    if (abortController) {
      abortController.abort()
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
      plugin: null
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
    isContinuation: boolean = false
  ) => {
    try {
      setUserInput("")
      setIsGenerating(true)
      setIsPromptPickerOpen(false)
      setIsAtPickerOpen(false)
      setNewMessageImages([])

      const newAbortController = new AbortController()
      setAbortController(newAbortController)

      const modelData = [
        ...models.map(model => ({
          modelId: model.model_id as LLMID,
          modelName: model.name,
          provider: "custom" as ModelProvider,
          hostedId: model.id,
          platformLink: "",
          imageInput: false
        })),
        ...LLM_LIST,
        ...availableLocalModels,
        ...availableOpenRouterModels
      ].find(llm => llm.modelId === chatSettings?.model)

      validateChatSettings(
        chatSettings,
        modelData,
        profile,
        selectedWorkspace,
        isContinuation,
        messageContent
      )

      if (!isContinuation && messageContent) {
        const urlRegex = /https?:\/\/[^\s]+/g
        const urls = messageContent.match(urlRegex) || []

        if (selectedPlugin !== PluginID.WEB_SCRAPER && urls.length > 0) {
          if (selectedPlugin === PluginID.NONE) {
            toast.warning("Enable the Web Scraper plugin to process websites.")
          }
        } else {
          for (const url of urls) {
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
              if (fileFromDb) {
                if (
                  !newMessageFiles.some(file => file.id === fileFromDb.id) &&
                  !chatFiles.some(file => file.id === fileFromDb.id)
                ) {
                  handleSelectUserFile(fileFromDb)
                  setFiles(prevFiles => [...prevFiles, fileFromDb])
                  newMessageFiles.push({ ...fileFromDb, file: null })
                }
              } else {
                toast.error("File not found in database.")
              }
            } else {
              toast.error("Failed to process websites.")
            }
          }
        }
      }
      let currentChat = selectedChat ? { ...selectedChat } : null

      const b64Images = newMessageImages.map(image => image.base64)

      let retrievedFileItems: Tables<"file_items">[] = []

      if (
        (newMessageFiles.length > 0 || chatFiles.length > 0) &&
        useRetrieval &&
        !isContinuation
      ) {
        setToolInUse("retrieval")

        retrievedFileItems = await handleRetrieval(
          userInput,
          newMessageFiles,
          chatFiles,
          chatSettings!.embeddingsProvider,
          sourceCount
        )
      }

      const lastMessageId =
        chatMessages.length > 0
          ? chatMessages[chatMessages.length - 1].message.id
          : null

      const { tempUserChatMessage, tempAssistantChatMessage } =
        createTempMessages(
          messageContent,
          chatMessages,
          chatSettings!,
          b64Images,
          isRegeneration,
          isContinuation,
          setChatMessages
        )

      const sentChatMessages = [...chatMessages]

      if (!isRegeneration) {
        sentChatMessages.push(tempUserChatMessage)
        if (!isContinuation) sentChatMessages.push(tempAssistantChatMessage)
      } else {
        sentChatMessages.pop()
        sentChatMessages.push(tempAssistantChatMessage)
      }

      let payload: ChatPayload = {
        chatSettings: chatSettings!,
        workspaceInstructions: selectedWorkspace!.instructions || "",
        chatMessages: sentChatMessages,
        assistant: selectedChat?.assistant_id ? selectedAssistant : null,
        messageFileItems: retrievedFileItems
      }

      let generatedText = ""
      let finishReasonFromResponse = ""

      // if (!isContinuation && selectedTools.length > 0) {
      //   setToolInUse(selectedTools.length > 1 ? "Tools" : selectedTools[0].name)

      //   const formattedMessages = await buildFinalMessages(
      //     payload,
      //     profile!,
      //     chatImages,
      //     selectedPlugin
      //   )

      //   const response = await fetch("/api/chat/tools", {
      //     method: "POST",
      //     headers: {
      //       "Content-Type": "application/json"
      //     },
      //     body: JSON.stringify({
      //       chatSettings: payload.chatSettings,
      //       messages: formattedMessages,
      //       selectedTools
      //     })
      //   })

      //   setToolInUse("none")

      //   const { fullText, finishReason } = await processResponse(
      //     response,
      //     isRegeneration
      //       ? payload.chatMessages[payload.chatMessages.length - 1]
      //       : tempAssistantChatMessage,
      //     newAbortController,
      //     setFirstTokenReceived,
      //     setChatMessages,
      //     setToolInUse
      //   )
      //   generatedText = fullText
      //   finishReasonFromResponse = finishReason
      if (
        selectedPlugin.length > 0 &&
        selectedPlugin !== PluginID.NONE &&
        selectedPlugin !== PluginID.WEB_SCRAPER
      ) {
        let fileContent = ""
        let fileName = ""

        if (newMessageFiles.length > 0 && newMessageFiles[0].type === "text") {
          const fileId = newMessageFiles[0].id
          const response = await fetch(`/api/retrieval/file/`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json"
            },
            body: JSON.stringify({ fileId: fileId })
          })
          const fileData = await response.json()
          fileContent = fileData.content
          fileName = newMessageFiles[0].name
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
          fileContent,
          fileName
        )
        generatedText = fullText
        finishReasonFromResponse = finishReason
      } else {
        const { fullText, finishReason } = await handleHostedChat(
          payload,
          profile!,
          modelData!,
          tempAssistantChatMessage,
          isRegeneration,
          isContinuation,
          newAbortController,
          newMessageImages,
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
      }

      if (!currentChat) {
        currentChat = await handleCreateChat(
          chatSettings!,
          profile!,
          selectedWorkspace!,
          messageContent || "",
          selectedAssistant!,
          newMessageFiles,
          finishReasonFromResponse,
          setSelectedChat,
          setChats,
          setChatFiles
        )
      } else {
        const updatedChat = await updateChat(currentChat.id, {
          updated_at: new Date().toISOString(),
          finish_reason: finishReasonFromResponse
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
        setChatImages
      )

      setIsGenerating(false)
      setFirstTokenReceived(false)
      setUserInput("")
    } catch (error) {
      setIsGenerating(false)
      setFirstTokenReceived(false)
    }
  }

  const handleSendEdit = async (
    editedContent: string,
    sequenceNumber: number
  ) => {
    if (!selectedChat) return

    await deleteMessagesIncludingAndAfter(
      selectedChat.user_id,
      selectedChat.id,
      sequenceNumber
    )

    const filteredMessages = chatMessages.filter(
      chatMessage => chatMessage.message.sequence_number < sequenceNumber
    )

    setChatMessages(filteredMessages)

    handleSendMessage(editedContent, filteredMessages, false)
  }

  return {
    chatInputRef,
    prompt,
    handleNewChat,
    handleSendMessage,
    handleFocusChatInput,
    handleStopMessage,
    handleSendContinuation,
    handleSendEdit,
    handleSendFeedback
  }
}
