import React, { useState, useCallback, useEffect, useMemo } from "react"
import {
  AgentMultibandAudioVisualizer,
  VisualizerState
} from "./agent-multiband-audio-visualizer"
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
  IconX,
  IconMicrophone,
  IconMicrophoneOff
} from "@tabler/icons-react"
import { AudioInputTile } from "./audioInput-tile"

interface ActiveRoomProps {
  onClose: () => void
}

const ActiveRoom: React.FC<ActiveRoomProps> = ({ onClose }) => {
  const { localParticipant } = useLocalParticipant()
  const [isPaused, setIsPaused] = useState(false)
  const [isMicMuted, setIsMicMuted] = useState(false)
  const [visualizerState, setVisualizerState] =
    useState<VisualizerState>("speaking")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const participants = useParticipants()
  const agentParticipant = participants.find(p => p.isAgent)
  const isAgentConnected = agentParticipant !== undefined
  const isSpeaking = useIsSpeaking(participants[0])

  const roomState = useConnectionState()
  const tracks = useTracks()

  useEffect(() => {
    const enableMicrophone = async () => {
      try {
        await localParticipant?.setMicrophoneEnabled(true)
        if (!loading) {
          setError(null)
        }
      } catch {
        if (!loading) {
          setError(
            "Microphone permission denied. Please enable the microphone."
          )
        }
      }
    }

    if (roomState === ConnectionState.Connected) {
      enableMicrophone()
    }
  }, [localParticipant, roomState, loading])

  const localTracks = tracks.filter(
    ({ participant }) => participant instanceof LocalParticipant
  )

  const agentAudioTrack = tracks.find(
    trackRef =>
      trackRef.publication.kind === Track.Kind.Audio &&
      trackRef.participant.isAgent
  )

  const subscribedVolumes = useMultibandTrackVolume(
    agentAudioTrack?.publication.track,
    5
  )

  const localMicTrack = localTracks.find(
    ({ source }) => source === Track.Source.Microphone
  )

  const localMultibandVolume = useMultibandTrackVolume(
    localMicTrack?.publication.track,
    20
  )

  const handleClose = useCallback(() => {
    setIsPaused(false)
    setVisualizerState("idle")
    onClose()
  }, [onClose])

  const handlePause = useCallback(() => {
    setIsPaused(prev => !prev)
  }, [])

  const handleMicMute = useCallback(() => {
    setIsMicMuted(prev => !prev)
  }, [])

  useEffect(() => {
    setVisualizerState(
      loading ? "thinking" : isSpeaking ? "listening" : "speaking"
    )
  }, [loading, isSpeaking])

  useEffect(() => {
    setLoading(tracks.length === 0 || !isAgentConnected || !agentAudioTrack)
  }, [tracks, isAgentConnected, agentAudioTrack])

  useEffect(() => {
    if (!loading) {
      setError(agentAudioTrack ? null : "Agent audio track not found")
    }
  }, [agentAudioTrack, loading])

  return (
    <>
      <div className="flex size-full items-center justify-center">
        <RoomAudioRenderer />

        <AgentMultibandAudioVisualizer
          state={visualizerState}
          barWidth={40}
          minBarHeight={40}
          maxBarHeight={200}
          accentColor="primary"
          frequencies={subscribedVolumes}
          borderRadius={12}
          gap={16}
        />
      </div>
      {loading && (
        <div className="absolute inset-0 top-1/3 flex items-center justify-center p-4">
          <div className="p-4">Loading...</div>
        </div>
      )}
      {error && (
        <div
          className="absolute inset-0 flex items-center justify-center p-4"
          style={{ top: "50%" }}
        >
          <div className="p-4">Error: {error}</div>
        </div>
      )}
      <button
        onClick={handlePause}
        className="bg-primary text-secondary absolute bottom-10 left-8 rounded-full p-4 shadow-lg hover:opacity-50"
      >
        {isPaused ? (
          <IconPlayerPlayFilled size={32} />
        ) : (
          <IconPlayerPauseFilled size={32} />
        )}
      </button>
      <button
        onClick={handleMicMute}
        className={`text-primary ${isMicMuted ? "opacity-50 hover:opacity-100" : "opacity-100 hover:opacity-50"} absolute bottom-10 left-1/2 flex -translate-x-1/2 flex-col items-center p-4`}
      >
        <div className="flex flex-row items-center gap-1">
          {isMicMuted ? (
            <IconMicrophoneOff size={20} />
          ) : (
            <IconMicrophone size={20} />
          )}
          {localMicTrack && (
            <AudioInputTile frequencies={localMultibandVolume} />
          )}
        </div>
      </button>
      <button
        onClick={handleClose}
        className="bg-primary text-secondary absolute bottom-10 right-8 rounded-full p-4 shadow-lg hover:opacity-50"
      >
        <IconX size={32} strokeWidth={3} />
      </button>
    </>
  )
}

export default ActiveRoom
