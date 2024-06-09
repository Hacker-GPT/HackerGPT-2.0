import { useEffect, useState } from "react"

export type VisualizerState = "listening" | "idle" | "speaking" | "thinking"

type AgentMultibandAudioVisualizerProps = {
  state: VisualizerState
  barWidth: number
  minBarHeight: number
  maxBarHeight: number
  accentColor: string
  accentShade?: number
  frequencies: Float32Array[]
  borderRadius: number
  gap: number
  userMic?: boolean
}

export const AgentMultibandAudioVisualizer = ({
  state,
  barWidth,
  minBarHeight,
  maxBarHeight,
  accentColor = "primary",
  accentShade,
  frequencies,
  borderRadius,
  gap,
  userMic
}: AgentMultibandAudioVisualizerProps) => {
  const summedFrequencies = frequencies.map(bandFrequencies => {
    const sum = bandFrequencies.reduce((a, b) => a + b, 0)
    return Math.sqrt(sum / bandFrequencies.length)
  })

  const [thinkingIndex, setThinkingIndex] = useState(
    Math.floor(summedFrequencies.length / 2)
  )
  const [thinkingDirection, setThinkingDirection] = useState<"left" | "right">(
    "right"
  )

  useEffect(() => {
    if (state !== "thinking") {
      setThinkingIndex(Math.floor(summedFrequencies.length / 2))
      return
    }
    const timeout = setTimeout(() => {
      if (thinkingDirection === "right") {
        if (thinkingIndex === summedFrequencies.length - 1) {
          setThinkingDirection("left")
          setThinkingIndex(prev => prev - 1)
        } else {
          setThinkingIndex(prev => prev + 1)
        }
      } else {
        if (thinkingIndex === 0) {
          setThinkingDirection("right")
          setThinkingIndex(prev => prev + 1)
        } else {
          setThinkingIndex(prev => prev - 1)
        }
      }
    }, 200)

    return () => clearTimeout(timeout)
  }, [state, summedFrequencies.length, thinkingDirection, thinkingIndex])

  const displayedFrequencies = userMic
    ? summedFrequencies.filter((_, index) => index % 5 === 0)
    : summedFrequencies

  return (
    <div
      className={`flex flex-row items-center`}
      style={{
        gap: gap + "px"
      }}
    >
      {displayedFrequencies.map((frequency, index) => {
        const isCenter = index === Math.floor(displayedFrequencies.length / 2)
        let color = accentShade ? `${accentColor}-${accentShade}` : accentColor
        let shadow = `shadow-lg-${accentColor}`
        let transform

        if (state === "listening" || state === "idle") {
          color = isCenter ? color : "gray-950"
          shadow = isCenter ? shadow : ""
          transform = isCenter ? "scale(1.2)" : "scale(1.0)"
        } else if (state === "speaking") {
          color = color
        } else if (state === "thinking") {
          color = index === thinkingIndex ? color : "gray-950"
          shadow = ""
          transform = index === thinkingIndex ? "scale(1.1)" : "scale(1)"
        }

        return (
          <div
            className={`bg-${color} ${shadow} ${
              isCenter && state === "listening" ? "animate-pulse" : ""
            }`}
            key={"frequency-" + index}
            style={{
              height:
                minBarHeight +
                frequency *
                  (maxBarHeight - minBarHeight) *
                  (userMic ? 0.5 : 1) +
                "px",
              borderRadius: borderRadius + "px",
              width: barWidth + "px",
              transition:
                "background-color 0.35s ease-out, transform 0.25s ease-out",
              transform: transform
            }}
          ></div>
        )
      })}
    </div>
  )
}
