import { useState, useEffect } from "react"

const useSpeechRecognition = () => {
  const [isListening, setIsListening] = useState(false)
  const [transcript, setTranscript] = useState("")
  const [isSupported, setIsSupported] = useState(true)

  useEffect(() => {
    if (!("webkitSpeechRecognition" in window)) {
      console.error("Speech recognition not supported in this browser.")
      setIsSupported(false)
      return
    }

    const recognition = new (window as any).webkitSpeechRecognition()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = "en-US"

    recognition.onstart = () => {
      console.log("Speech recognition started")
      setIsListening(true)
    }
    recognition.onend = () => {
      console.log("Speech recognition ended")
      setIsListening(false)
    }
    recognition.onerror = (event: any) => {
      console.error("Speech recognition error:", event.error)
      if (event.error === "network") {
        console.warn("Ignoring network error during speech recognition.")
        // Optionally, you can restart recognition here if needed
        recognition.start()
      } else {
        setIsListening(false)
      }
    }

    recognition.onresult = (event: any) => {
      let interimTranscript = ""
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          setTranscript(prev => prev + event.results[i][0].transcript)
        } else {
          interimTranscript += event.results[i][0].transcript
        }
      }
      console.log("Transcript:", interimTranscript)
    }

    if (isListening) {
      recognition.start()
    } else {
      recognition.stop()
    }

    return () => {
      recognition.stop()
    }
  }, [isListening])

  const startListening = () => {
    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then(() => setIsListening(true))
      .catch(err => {
        console.error("Microphone access denied:", err)
        setIsListening(false)
      })
  }

  return {
    isListening,
    transcript,
    setIsListening,
    isSupported,
    startListening
  }
}

export default useSpeechRecognition
