import React, { useState, useCallback } from "react"
import {
  RoomAudioRenderer,
  useLocalParticipant
} from "@livekit/components-react"

const ActiveRoom: React.FC = () => {
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant()
  const [isPaused, setIsPaused] = useState(false)

  const togglePause = useCallback(() => {
    localParticipant?.setMicrophoneEnabled(!isPaused)
    setIsPaused(!isPaused)
  }, [isPaused, localParticipant])

  return (
    <>
      <RoomAudioRenderer />
      <button
        onClick={() =>
          localParticipant?.setMicrophoneEnabled(!isMicrophoneEnabled)
        }
      >
        Toggle Microphone
      </button>
      <div>Audio Enabled: {isMicrophoneEnabled ? "Muted" : "Unmuted"}</div>
      <button onClick={togglePause}>
        {isPaused ? "Resume Conversation" : "Pause Conversation"}
      </button>
    </>
  )
}

export default ActiveRoom
