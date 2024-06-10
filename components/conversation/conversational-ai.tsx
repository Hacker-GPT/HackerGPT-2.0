import React, { useState, useEffect, useCallback, useMemo } from "react"
import { LiveKitRoom } from "@livekit/components-react"
import ActiveRoom from "./active-room"
import { fetchToken } from "./fetch-livekit-token"

interface ConversationalAIProps {
  onClose: () => void
}

const ConversationalAI: React.FC<ConversationalAIProps> = ({ onClose }) => {
  const [token, setToken] = useState<string | null>(null)
  const [url, setUrl] = useState<string | undefined>(undefined)
  const [loading, setLoading] = useState<boolean>(true)
  const [error, setError] = useState<string | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  const fetchTokenCallback = useCallback(async () => {
    const { token, url, error } = await fetchToken()
    setToken(token)
    setUrl(url)
    setError(error)
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchTokenCallback()
  }, [fetchTokenCallback])

  const handleOnClose = useCallback(() => {
    setIsConnected(false)
    onClose()
  }, [onClose])

  const handleOnConnected = useCallback(() => {
    setIsConnected(true)
  }, [])

  const handleOnDisconnected = useCallback(() => {
    setIsConnected(false)
  }, [])

  const content = useMemo(() => {
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
          onConnected={handleOnConnected}
          onDisconnected={handleOnDisconnected}
        >
          <ActiveRoom onClose={handleOnClose} />
        </LiveKitRoom>
      )
    }
    return null
  }, [
    loading,
    error,
    token,
    url,
    handleOnClose,
    handleOnConnected,
    handleOnDisconnected
  ])

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-[#fafafa] dark:bg-[#181818]">
      {content}
    </div>
  )
}

export default ConversationalAI