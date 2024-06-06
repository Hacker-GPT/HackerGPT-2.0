import React, { useState, useEffect, useCallback, useMemo } from "react"
import { IconX } from "@tabler/icons-react"
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

  const handleClose = useCallback(() => {
    setToken(null)
    setUrl(undefined)
    setLoading(true)
    setError(null)
    onClose()
  }, [onClose])

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
          token={token}
          serverUrl={url}
          connectOptions={{ autoSubscribe: true }}
        >
          <ActiveRoom />
        </LiveKitRoom>
      )
    }
    return null
  }, [loading, error, token, url])

  return (
    <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black bg-opacity-75">
      <div className="absolute right-4 top-4">
        <IconX
          className="cursor-pointer text-white hover:opacity-75"
          size={30}
          onClick={handleClose}
        />
      </div>
      <div className="flex size-full items-center justify-center">
        {content}
      </div>
    </div>
  )
}

export default ConversationalAI
