import "server-only"
import { CodeInterpreter } from "@e2b/code-interpreter"

const sandboxTimeout = 1 * 60 * 1000 // 1 minutes in ms
const template = "code-interpreter-stateful"

export async function executeCode(
  sessionID: string,
  code: string,
  userID: string
): Promise<{
  results: string | null
  stdout: string
  stderr: string
  error: string | null
}> {
  const sandbox = await getSandbox(sessionID, userID)

  try {
    const execution = await sandbox.notebook.execCell(code, {
      timeout: 55000
    })

    if (execution.error) {
      console.error(`[${sessionID}] Execution error:`, execution.error)
    }

    let formattedResults = null
    if (execution.results && execution.results.length > 0) {
      formattedResults = execution.results
        .map(result => (result.text ? result.text : JSON.stringify(result)))
        .join("\n")
    }

    return {
      results: formattedResults,
      stdout: execution.logs.stdout.join("\n"),
      stderr: execution.logs.stderr.join("\n"),
      error: execution.error ? formatFullError(execution.error) : null
    }
  } catch (error: any) {
    console.error(`[${sessionID}] Error in executeCode:`, error)

    return {
      results: null,
      stdout: "",
      stderr: "",
      error: formatFullError(error)
    }
  } finally {
    try {
      await sandbox.keepAlive(sandboxTimeout)
    } catch (keepAliveError) {
      console.warn(
        `[${sessionID}] Error keeping sandbox alive:`,
        keepAliveError
      )
    }
  }
}

function formatFullError(error: any): string {
  if (!error) return ""

  let output = ""
  if (error.name) output += `${error.name}: `
  if (error.value) output += `${error.value}\n\n`
  if (error.tracebackRaw && Array.isArray(error.tracebackRaw)) {
    output += error.tracebackRaw.join("\n")
  }
  return output.trim()
}

const sandboxCache = new Map<string, CodeInterpreter>()

async function getSandbox(
  sessionID: string,
  userID: string
): Promise<CodeInterpreter> {
  const E2B_API_KEY = process.env.E2B_API_KEY

  if (sandboxCache.has(sessionID)) {
    return sandboxCache.get(sessionID)!
  }

  const sandboxes = await CodeInterpreter.list()
  const existingSandbox = sandboxes.find(
    sandbox => sandbox.metadata?.sessionID === sessionID
  )

  let sandbox: CodeInterpreter
  if (existingSandbox) {
    sandbox = await CodeInterpreter.reconnect({
      sandboxID: existingSandbox.sandboxID,
      apiKey: E2B_API_KEY
    })
  } else {
    sandbox = await CodeInterpreter.create({
      template,
      apiKey: E2B_API_KEY,
      metadata: { sessionID, userID }
    })
  }

  sandboxCache.set(sessionID, sandbox)
  return sandbox
}

export async function closeSandbox(sessionID: string) {
  const sandbox = sandboxCache.get(sessionID)
  if (sandbox) {
    await sandbox.close().catch(console.error)
    sandboxCache.delete(sessionID)
  }
}
