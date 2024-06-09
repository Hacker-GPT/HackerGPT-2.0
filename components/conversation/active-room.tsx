import React, { useState, useCallback, useEffect, useMemo } from "react"
import {
  AgentMultibandAudioVisualizer,
  VisualizerState
} from "./agent-multiband-audio-visualizer"
import {
  RoomAudioRenderer,
  useLocalParticipant,
  useTracks,
  useParticipants,
  useIsSpeaking
} from "@livekit/components-react"
import { useMultibandTrackVolume } from "@/lib/hooks/useTrackVolume"
import { Track, createLocalAudioTrack, LocalParticipant } from "livekit-client"
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
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant()
  const [isPaused, setIsPaused] = useState(false)
  const [isMicMuted, setIsMicMuted] = useState(false)
  const [visualizerState, setVisualizerState] =
    useState<VisualizerState>("speaking")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const tracks = useTracks()
  const participants = useParticipants()
  const agentParticipant = participants.find(p => p.isAgent)
  const isAgentConnected = agentParticipant !== undefined
  const isSpeaking = useIsSpeaking(participants[0])

  const localTracks = useMemo(
    () =>
      tracks.filter(
        ({ participant }) => participant instanceof LocalParticipant
      ),
    [tracks]
  )

  const agentAudioTrack = useMemo(
    () =>
      tracks.find(
        trackRef =>
          trackRef.publication.kind === Track.Kind.Audio &&
          trackRef.participant.isAgent
      ),
    [tracks]
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
    setLoading(tracks.length === 0 || !isAgentConnected)
  }, [tracks, isAgentConnected])

  useEffect(() => {
    if (!loading) {
      setError(agentAudioTrack ? null : "Agent audio track not found")
    }
  }, [agentAudioTrack, loading])

  useEffect(() => {
    const enableMicrophone = async () => {
      try {
        await localParticipant?.setMicrophoneEnabled(true)
        if (!localMicTrack) {
          const audioTrack = await createLocalAudioTrack()
          await localParticipant?.publishTrack(audioTrack)
        }
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

    if (!isMicrophoneEnabled) {
      enableMicrophone()
    } else if (!loading) {
      setError(null)
    }
  }, [isMicrophoneEnabled, localParticipant, localTracks, loading])

  return (
    <>
      <div className="flex size-full items-center justify-center">
        <div className="relative flex items-center justify-center">
          {/* ## TODO: muted={false} on togglePause */}
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
      </div>
      {loading && (
        <div
          className="absolute inset-0 flex items-center justify-center p-4"
          style={{ top: "30%" }}
        >
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
        className="bg-primary text-secondary absolute bottom-10 left-10 rounded-full p-4 shadow-lg hover:opacity-50"
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
            <IconMicrophoneOff size={24} />
          ) : (
            <IconMicrophone size={24} />
          )}
          {localMicTrack && (
            <AudioInputTile frequencies={localMultibandVolume} />
          )}
        </div>
      </button>
      <button
        onClick={handleClose}
        className="bg-primary text-secondary absolute bottom-10 right-10 rounded-full p-4 shadow-lg hover:opacity-50"
      >
        <IconX size={32} strokeWidth={3} />
      </button>
    </>
  )
}

export default ActiveRoom
