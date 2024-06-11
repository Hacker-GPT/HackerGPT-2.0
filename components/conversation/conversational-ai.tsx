import React, { useState, useEffect, useCallback, useMemo } from "react"
import { LiveKitRoom } from "@livekit/components-react"
import ActiveRoom from "./active-room"
import { fetchToken } from "./fetch-livekit-token"
import { IconPlayerPauseFilled, IconX } from "@tabler/icons-react"

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
    setTryToConnect(false)
    setConnected(false)
    onClose()
  }, [onClose])

  const handleOnError = useCallback((error: Error) => {
    setState(prevState => ({ ...prevState, error: error.message }))
  }, [])

  const [tryToConnect, setTryToConnect] = useState(true)
  const [connected, setConnected] = useState(false)

  const handleRetry = useCallback(() => {
    setState(prevState => ({ ...prevState, loading: true, error: null }))
    setTryToConnect(true)
    fetchTokenCallback()
  }, [fetchTokenCallback])

  const content = useMemo(() => {
    const { loading, error, token, url } = state
    if (loading) {
      return <LoadingMessage />
    }
    if (error) {
      return <ErrorMessage />
    }
    if (token && url) {
      return (
        <LiveKitRoom
          video={false}
          audio={true}
          token={token}
          serverUrl={url}
          connect={tryToConnect}
          connectOptions={{ autoSubscribe: true }}
          onConnected={() => setConnected(true)}
          onDisconnected={() => {
            setTryToConnect(false)
            setConnected(false)
          }}
          onError={handleOnError}
        >
          <ActiveRoom />
        </LiveKitRoom>
      )
    }
    return null
  }, [state, handleOnClose, handleOnError, tryToConnect])

  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-[#fafafa] dark:bg-[#181818]"
      onClick={state.error ? handleRetry : undefined}
    >
      {content}
      {(state.error || state.loading) && (
        <button
          className="bg-primary text-secondary absolute bottom-10 left-8 rounded-full p-4 shadow-lg disabled:cursor-not-allowed disabled:opacity-50 md:hover:opacity-50"
          disabled={true}
        >
          <IconPlayerPauseFilled size={32} />
        </button>
      )}
      <button
        onClick={handleOnClose}
        className="bg-primary text-secondary absolute bottom-10 right-8 rounded-full p-4 shadow-lg md:hover:opacity-50"
      >
        <IconX size={32} strokeWidth={3} />
      </button>
    </div>
  )
}

const LoadingMessage = () => (
  <div className="absolute inset-0 top-1/2 flex items-center justify-center p-4">
    Connecting...
  </div>
)

const ErrorMessage = () => (
  <div className="absolute inset-0 top-1/2 flex items-center justify-center p-4">
    Connection failed, tap to retry
  </div>
)

export default ConversationalAI
