import llmConfig from "@/lib/models/llm/llm-config"
import { getServerProfileWithUserRole } from "@/lib/server/server-chat-helpers"

export async function GET(request: Request) {
  const url = new URL(request.url)
  const id = url.searchParams.get("id")

  if (!llmConfig.hackerRAG.getDataEndpoint)
    return new Response(JSON.stringify({ error: "No endpoint provided" }), {
      status: 500
    })
  if (!id)
    return new Response(JSON.stringify({ error: "No id provided" }), {
      status: 400
    })

  const profile = await getServerProfileWithUserRole()

  if (!profile || !profile.user_role || profile.user_role.role !== "moderator")
    return new Response(JSON.stringify({ error: "Unauthorized" }), {
      status: 401
    })

  const response = await fetch(
    llmConfig.hackerRAG.getDataEndpoint + `?id=${id}`,
    {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${llmConfig.hackerRAG.apiKey}`
      }
    }
  )

  if (!response.ok)
    return new Response(JSON.stringify({ error: "Failed to fetch data" }), {
      status: response.status
    })

  const data = await response.json()

  return new Response(JSON.stringify(data.result))
}
