import { supabase } from "@/lib/supabase/browser-client"
import { TablesUpdate } from "@/supabase/types"

export const getToolWorkspacesByWorkspaceId = async (workspaceId: string) => {
  const { data: workspace, error } = await supabase
    .from("workspaces")
    .select(
      `
      id,
      name,
      tools (*)
    `
    )
    .eq("id", workspaceId)
    .single()

  if (!workspace) {
    throw new Error(error.message)
  }

  return workspace
}

export const updateTool = async (
  toolId: string,
  tool: TablesUpdate<"tools">
) => {
  const { data: updatedTool, error } = await supabase
    .from("tools")
    .update(tool)
    .eq("id", toolId)
    .select("*")
    .single()

  if (error) {
    throw new Error(error.message)
  }

  return updatedTool
}
