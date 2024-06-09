import { AgentMultibandAudioVisualizer } from "./agent-multiband-audio-visualizer"

type AudioInputTileProps = {
  frequencies: Float32Array[]
}

export const AudioInputTile = ({ frequencies }: AudioInputTileProps) => {
  return (
    <div className={`w-full items-center justify-center rounded-sm`}>
      <AgentMultibandAudioVisualizer
        state="speaking"
        barWidth={20}
        minBarHeight={20}
        maxBarHeight={100}
        accentColor={"primary"}
        frequencies={frequencies}
        borderRadius={2}
        gap={4}
        userMic={true}
      />
    </div>
  )
}
