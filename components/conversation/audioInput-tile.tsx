import { AgentMultibandAudioVisualizer } from "./agent-multiband-audio-visualizer"

type AudioInputTileProps = {
  frequencies: Float32Array[]
}

export const AudioInputTile = ({ frequencies }: AudioInputTileProps) => {
  return (
    <div className={`w-full items-center justify-center rounded-sm`}>
      <AgentMultibandAudioVisualizer
        state="speaking"
        barWidth={16}
        minBarHeight={16}
        maxBarHeight={80}
        accentColor={"primary"}
        frequencies={frequencies}
        borderRadius={12}
        gap={4}
        userMic={true}
      />
    </div>
  )
}
