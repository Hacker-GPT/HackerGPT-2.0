import {
  AgentMultibandAudioVisualizer,
  VisualizerState
} from "./agent-multiband-audio-visualizer"

type AudioVisualizerTileProps = {
  frequencies: Float32Array[]
  source:
    | "user"
    | "agent"
    | "agent-desktop"
    | "agent-paused"
    | "agent-desktop-paused"
  visualizerState?: VisualizerState
}

export const AudioVisualizerTile = ({
  frequencies,
  source,
  visualizerState
}: AudioVisualizerTileProps) => {
  const config = {
    user: {
      state: "speaking" as VisualizerState,
      barWidth: 16,
      minBarHeight: 16,
      maxBarHeight: 80,
      accentColor: "primary",
      borderRadius: 12,
      gap: 4,
      userMic: true
    },
    agent: {
      state: visualizerState as VisualizerState,
      barWidth: 40,
      minBarHeight: 40,
      maxBarHeight: 200,
      accentColor: "primary",
      borderRadius: 12,
      gap: 16,
      userMic: false
    },
    "agent-desktop": {
      state: visualizerState as VisualizerState,
      barWidth: 80,
      minBarHeight: 80,
      maxBarHeight: 320,
      accentColor: "primary",
      borderRadius: 12,
      gap: 16,
      userMic: false
    },
    "agent-paused": {
      state: "idle" as VisualizerState,
      barWidth: 180,
      minBarHeight: 180,
      maxBarHeight: 180,
      accentColor: "primary",
      borderRadius: 12,
      gap: 16,
      userMic: false
    },
    "agent-desktop-paused": {
      state: "idle" as VisualizerState,
      barWidth: 240,
      minBarHeight: 240,
      maxBarHeight: 240,
      accentColor: "primary",
      borderRadius: 12,
      gap: 16,
      userMic: false
    }
  }[source]

  return (
    <div
      className={
        source === "user" ? `w-full items-center justify-center rounded-sm` : ``
      }
    >
      <AgentMultibandAudioVisualizer
        state={config.state}
        barWidth={config.barWidth}
        minBarHeight={config.minBarHeight}
        maxBarHeight={config.maxBarHeight}
        accentColor={config.accentColor}
        frequencies={frequencies}
        borderRadius={config.borderRadius}
        gap={config.gap}
        userMic={config.userMic}
      />
    </div>
  )
}
