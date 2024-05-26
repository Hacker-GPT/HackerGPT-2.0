import { useState, useEffect, useRef } from "react"
import { toast } from "sonner"

const useSpeechRecognition = (
  onTranscriptChange: (transcript: string) => void
) => {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [isSupported, setIsSupported] = useState(true)
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null)
  const audioChunksRef = useRef<Blob[]>([])
  const isCanceledRef = useRef(false)

  useEffect(() => {
    if (!navigator.mediaDevices || !window.MediaRecorder) {
      setIsSupported(false)
      return
    }

    const handleDataAvailable = (event: BlobEvent) => {
      if (event.data.size > 0) {
        audioChunksRef.current.push(event.data)
      }
    }

    const handleStop = async () => {
      if (isCanceledRef.current) {
        audioChunksRef.current = []
        isCanceledRef.current = false
        return
      }

      const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" })

      const formData = new FormData()
      formData.append("audioFile", audioBlob, "speech.webm")

      try {
        const response = await fetch("/api/v2/chat/speech-to-text", {
          method: "POST",
          body: formData
        })

        if (!response.ok) {
          throw new Error("Failed to transcribe audio")
        }

        const data = await response.json()
        setTranscript(data.text)
        onTranscriptChange(data.text)
      } catch (error) {
        toast.error("Error transcribing audio: " + error)
      }

      audioChunksRef.current = []
    }

    const startRecording = () => {
      navigator.mediaDevices
        .getUserMedia({ audio: true })
        .then(stream => {
          const options = { mimeType: "audio/webm" }
          const recorder = new MediaRecorder(stream, options)
          recorder.ondataavailable = handleDataAvailable
          recorder.onstop = handleStop
          recorder.start()
          setMediaRecorder(recorder)
          setIsListening(true)
        })
        .catch(err => {
          toast.error("Microphone access denied: " + err)
          setIsListening(false)
        })
    }

    if (isListening) {
      startRecording()
    } else if (mediaRecorder) {
      mediaRecorder.stop()
    }

    return () => {
      if (mediaRecorder) {
        mediaRecorder.stop()
      }
    }
  }, [isListening])

  const startListening = () => {
    setIsListening(true)
  }

  const cancelListening = () => {
    isCanceledRef.current = true
    setIsListening(false)
  }

  return {
    isListening,
    transcript,
    setIsListening,
    isSupported,
    startListening,
    cancelListening
  }
}

export default useSpeechRecognition
