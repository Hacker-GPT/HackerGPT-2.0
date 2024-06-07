import React, { useState, useCallback, useEffect, useMemo } from "react"
import {
  AgentMultibandAudioVisualizer,
  VisualizerState
} from "./agent-multiband-audio-visualizer"
import {
  RoomAudioRenderer,
  useLocalParticipant,
  useTracks,
  useParticipants
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

interface ActiveRoomProps {
  onClose: () => void
}

const ActiveRoom: React.FC<ActiveRoomProps> = ({ onClose }) => {
  const { localParticipant, isMicrophoneEnabled } = useLocalParticipant()
  const [isPaused, setIsPaused] = useState(false)
  const [visualizerState, setVisualizerState] =
    useState<VisualizerState>("speaking")
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isMicMuted, setIsMicMuted] = useState(false)
  const tracks = useTracks()
  const participants = useParticipants()
  const agentParticipant = participants.find(p => p.isAgent)
  const isAgentConnected = agentParticipant !== undefined

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

  const handleClose = useCallback(() => {
    setIsPaused(false)
    setVisualizerState("idle")
    onClose()
  }, [onClose])

  useEffect(() => {
    setLoading(tracks.length === 0 || !isAgentConnected)
  }, [tracks, isAgentConnected])

  useEffect(() => {
    if (!agentAudioTrack) {
      setError("Agent audio track not found")
    } else {
      setError(null)
    }
  }, [agentAudioTrack])

  useEffect(() => {
    const enableMicrophone = async () => {
      try {
        await localParticipant?.setMicrophoneEnabled(true)
        const localMicTrack = localTracks.find(
          track => track.source === Track.Source.Microphone
        )
        if (!localMicTrack) {
          const audioTrack = await createLocalAudioTrack()
          await localParticipant?.publishTrack(audioTrack)
        }
        setError(null)
      } catch (err) {
        setError("Microphone permission denied. Please enable the microphone.")
      }
    }

    if (!isMicrophoneEnabled) {
      enableMicrophone()
    }
  }, [isMicrophoneEnabled, localParticipant, localTracks])

  return (
    <>
      <div className="flex size-full items-center justify-center">
        <div className="relative">
          {loading && <div>Loading...</div>}
          {error && <div>Error: {error}</div>}
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
      <button
        // TODO: onClick={togglePause}
        className="bg-primary text-secondary absolute bottom-10 left-10 rounded-full p-4 shadow-lg hover:opacity-50"
      >
        {isPaused ? (
          <IconPlayerPlayFilled size={32} />
        ) : (
          <IconPlayerPauseFilled size={32} />
        )}
      </button>
      <button
        // TODO: onClick={toggleMicMute}
        className={`text-primary ${isMicMuted ? "opacity-50 hover:opacity-100" : "opacity-100 hover:opacity-50"} absolute bottom-10 left-1/2 -translate-x-1/2 p-4`}
      >
        {isMicMuted ? (
          <IconMicrophoneOff size={32} />
        ) : (
          <IconMicrophone size={32} />
        )}
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
