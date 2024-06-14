import React, { useState, useCallback, useEffect } from "react"
import { VisualizerState } from "./agent-multiband-audio-visualizer"
import {
  useConnectionState,
  RoomAudioRenderer,
  useLocalParticipant,
  useTracks,
  useParticipants,
  useIsSpeaking
} from "@livekit/components-react"
import { useMultibandTrackVolume } from "@/lib/hooks/useTrackVolume"
import { Track, LocalParticipant, ConnectionState } from "livekit-client"
import {
  IconPlayerPauseFilled,
  IconPlayerPlayFilled,
  IconMicrophone,
  IconMicrophoneOff
} from "@tabler/icons-react"
import { AudioVisualizerTile } from "./audio-visualizer-tile"

const ActiveRoom: React.FC = () => {
  const { localParticipant } = useLocalParticipant()
  const [isPaused, setIsPaused] = useState(false)
  const [isMicMuted, setIsMicMuted] = useState(false)
  const [visualizerState, setVisualizerState] =
    useState<VisualizerState>("speaking")
  const [loading, setLoading] = useState(true)
  const [message, setMessage] = useState<string | null>(null)
  const participants = useParticipants()
  const agentParticipant = participants.find(p => p.isAgent)
  const isAgentConnected = agentParticipant !== undefined
  const isSpeaking = useIsSpeaking(participants[0])

  const roomState = useConnectionState()
  const tracks = useTracks()

  useEffect(() => {
    const enableMicrophone = async () => {
      await localParticipant?.setMicrophoneEnabled(true)
    }

    if (roomState === ConnectionState.Connected) {
      enableMicrophone()
    }
  }, [localParticipant, roomState])

  const localTracks = tracks.filter(
    ({ participant }) => participant instanceof LocalParticipant
  )

  const agentAudioTrack = tracks.find(
    trackRef =>
      trackRef.publication.kind === Track.Kind.Audio &&
      trackRef.participant.isAgent
  )

  const subscribedVolumes = useMultibandTrackVolume(
    agentAudioTrack?.publication.track
  )

  const localMicTrack = localTracks.find(
    ({ source }) => source === Track.Source.Microphone
  )

  const localMultibandVolume = useMultibandTrackVolume(
    localMicTrack?.publication.track,
    20
  )

  const handlePause = useCallback(() => {
    setIsPaused(prev => !prev)
    console.log("paused", isPaused)
    if (!isPaused) {
      setIsMicMuted(true)
      localParticipant?.setMicrophoneEnabled(false)
      agentAudioTrack?.publication.track?.stop()
    } else {
      setIsMicMuted(false)
      localParticipant?.setMicrophoneEnabled(true)
    }
    // TODO: stop/start the agent audio track
  }, [isPaused, localParticipant, agentAudioTrack])

  const handleMicMute = useCallback(() => {
    setIsMicMuted(prev => !prev)
    localParticipant?.setMicrophoneEnabled(isMicMuted)
  }, [isMicMuted, localParticipant])

  useEffect(() => {
    setVisualizerState(
      loading ? "thinking" : isSpeaking ? "listening" : "speaking"
    )
  }, [loading, isSpeaking])

  useEffect(() => {
    setLoading(tracks.length === 0 || !isAgentConnected || !agentAudioTrack)
  }, [tracks, isAgentConnected, agentAudioTrack])

  const handleScreenClick = useCallback(() => {
    if (isPaused) {
      handlePause()
    }
  }, [isPaused, handlePause])

  useEffect(() => {
    if (isPaused) {
      setMessage("Tap to activate")
      setVisualizerState("idle")
      document.addEventListener("click", handleScreenClick)
    } else {
      setMessage(null)
      setVisualizerState("speaking")
      document.removeEventListener("click", handleScreenClick)
    }

    return () => {
      document.removeEventListener("click", handleScreenClick)
    }
  }, [isPaused, handleScreenClick])

  return (
    <>
      <RoomAudioRenderer />
      <div className="absolute inset-0 bottom-1/4 flex items-center justify-center">
        {loading ? (
          <AudioVisualizerTile
            frequencies={localMultibandVolume}
            source="agent"
            visualizerState="thinking"
          />
        ) : (
          <AudioVisualizerTile
            frequencies={subscribedVolumes}
            source={isPaused ? "agent-paused" : "agent"}
            visualizerState={visualizerState}
          />
        )}
      </div>
      <div className="absolute inset-0 top-1/2 flex items-center justify-center p-4">
        {message ? (
          <div className="p-4">{message}</div>
        ) : (
          loading && <div className="p-4">Loading...</div>
        )}
      </div>
      <button
        onClick={handlePause}
        className="bg-primary text-secondary absolute bottom-10 left-8 rounded-full p-4 shadow-lg disabled:cursor-not-allowed disabled:opacity-50 md:hover:opacity-50"
        disabled={loading}
      >
        {isPaused ? (
          <IconPlayerPlayFilled size={32} />
        ) : (
          <IconPlayerPauseFilled size={32} />
        )}
      </button>
      {!isPaused && localMicTrack && !loading && (
        <button
          onClick={handleMicMute}
          className={`text-primary ${isMicMuted ? "opacity-50 md:hover:opacity-100" : "opacity-100 md:hover:opacity-50"} absolute bottom-10 left-1/2 flex -translate-x-1/2 flex-col items-center p-4`}
        >
          <div className="flex flex-row items-center gap-1">
            {isMicMuted ? (
              <IconMicrophoneOff size={20} />
            ) : (
              <IconMicrophone size={20} />
            )}
            <AudioVisualizerTile
              frequencies={localMultibandVolume}
              source="user"
            />
          </div>
        </button>
      )}
    </>
  )
}

export default ActiveRoom
