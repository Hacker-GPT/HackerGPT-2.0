export const fetchToken = async () => {
  try {
    const response = await fetch("/api/livekit-token")
    if (!response.ok) {
      throw new Error("Network response was not ok")
    }
    const { accessToken, url } = await response.json()
    return { token: accessToken, url, error: null }
  } catch (error) {
    console.error("Failed to fetch token:", error)
    return {
      token: null,
      url: undefined,
      error: "Failed to connect. Please try again."
    }
  }
}
