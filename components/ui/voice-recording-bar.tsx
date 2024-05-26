import React, { FC, useRef, useState } from "react"
import { IconX, IconCheck } from "@tabler/icons-react"

interface VoiceRecordingBarProps {
  isListening: boolean
  stopListening: () => void
  cancelListening: () => void
}

const VoiceRecordingBar: FC<VoiceRecordingBarProps> = ({
  isListening,
  stopListening,
  cancelListening
}) => {
  const [position, setPosition] = useState(0)
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const [isRecording, setIsRecording] = useState(false)

  const startRecording = () => {
    if (!isRecording) {
      setIsRecording(true)
      timerRef.current = setInterval(() => {
        setPosition(prevPosition => prevPosition + 1)
      }, 1000)
    }
  }

  const stopRecording = () => {
    if (isRecording) {
      setIsRecording(false)
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }

  const handleRecordingChange = (stop: boolean) => {
    stopRecording()
    if (stop) {
      stopListening()
    } else {
      cancelListening()
    }
  }

  if (isListening && !isRecording) {
    startRecording()
  } else if (!isListening && isRecording) {
    stopRecording()
  }

  return (
    <div className="mt-3 flex min-h-[60px] items-center justify-between rounded-xl border-2 border-gray-300 bg-transparent px-4 py-2">
      <IconX
        className="bg-primary text-secondary cursor-pointer rounded text-gray-500 hover:text-gray-700"
        onClick={() => handleRecordingChange(false)}
        size={24}
      />
      <div className="flex-1">
        <div className="mx-2">
          {isRecording && (
            <div className="flex items-center justify-center">
              <div className="size-3 animate-ping rounded-full bg-red-500"></div>
              <div className="ml-2 text-sm text-gray-500">Recording...</div>
            </div>
          )}
        </div>
      </div>
      <div className="flex items-center">
        <div className="mr-2 text-sm text-gray-500">{formatTime(position)}</div>
        <IconCheck
          className="bg-primary text-secondary cursor-pointer rounded p-1 hover:opacity-50"
          onClick={() => handleRecordingChange(true)}
          size={28}
        />
      </div>
    </div>
  )
}

function formatTime(time: number): string {
  const roundedTime = Math.floor(time)
  const minutes = Math.floor(roundedTime / 60)
  const seconds = roundedTime % 60
  return `${minutes}:${seconds < 10 ? "0" : ""}${seconds}`
}

export default VoiceRecordingBar
