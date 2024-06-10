import React, { useState, useEffect, useCallback, useMemo } from "react"
import { LiveKitRoom } from "@livekit/components-react"
import ActiveRoom from "./active-room"
import { fetchToken } from "./fetch-livekit-token"
import { IconX } from "@tabler/icons-react"

interface ConversationalAIProps {
  onClose: () => void
}

const ConversationalAI: React.FC<ConversationalAIProps> = ({ onClose }) => {
  const [state, setState] = useState({
    token: null as string | null,
    url: undefined as string | undefined,
    loading: true,
    error: null as string | null
  })

  const fetchTokenCallback = useCallback(async () => {
    const { token, url, error } = await fetchToken()
    setState({ token, url, error, loading: false })
  }, [])

  useEffect(() => {
    fetchTokenCallback()
  }, [fetchTokenCallback])

  const handleOnClose = useCallback(() => {
    onClose()
  }, [onClose])

  const handleOnError = useCallback((error: Error) => {
    setState(prevState => ({ ...prevState, error: error.message }))
  }, [])

  const content = useMemo(() => {
    const { loading, error, token, url } = state
    if (loading) {
      return <div className="text-white">Connecting...</div>
    }
    if (error) {
      return <div className="text-white">{error}</div>
    }
    if (token && url) {
      return (
        <LiveKitRoom
          video={false}
          audio={true}
          token={token}
          serverUrl={url}
          connectOptions={{ autoSubscribe: true }}
          onError={handleOnError}
        >
          <ActiveRoom onClose={handleOnClose} />
        </LiveKitRoom>
      )
    }
    return null
  }, [state, handleOnClose, handleOnError])

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-[#fafafa] dark:bg-[#181818]">
      {content}
      {state.error && (
        <button
          onClick={handleOnClose}
          className="bg-primary text-secondary absolute bottom-10 justify-center rounded-full p-4 shadow-lg hover:opacity-50"
        >
          <IconX size={32} strokeWidth={3} />
        </button>
      )}
    </div>
  )
}

export default ConversationalAI
