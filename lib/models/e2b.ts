import { CodeInterpreter } from "@e2b/code-interpreter"
import { Tool } from "ai"

const E2B_API_KEY = process.env.E2B_API_KEY
if (!E2B_API_KEY) {
  throw new Error("E2B_API_KEY environment variable not found")
}

const sandboxTimeout = 1 * 60 * 1000 // 1 minutes in ms
const template = "code-interpreter-stateful"

/**
 * Evaluate the code in the given session.
 * @param sessionID The session ID to evaluate the code in.
 * @param code The code to evaluate.
 * @returns The result of the evaluation.
 */
export async function evaluateCode(sessionID: string, code: string) {
  const sandbox = await getSandbox(sessionID)

  try {
    const execution = await sandbox.notebook.execCell(code, {})

    return {
      results: execution.results,
      stdout: execution.logs.stdout,
      stderr: execution.logs.stderr,
      error: execution.error
    }
  } finally {
    try {
      await sandbox.keepAlive(sandboxTimeout)
    } catch {
      // Ignore errors from the keepalive and close the sandbox
    }
    await sandbox.close()
  }
}

/**
 * Get the sandbox for the given session ID.
 * @param sessionID The session ID to get the sandbox for.
 * @returns The sandbox for the given session ID.
 */
async function getSandbox(sessionID: string) {
  const sandboxes = await CodeInterpreter.list()
  const sandboxID = sandboxes.find(
    sandbox => sandbox.metadata?.sessionID === sessionID
  )?.sandboxID

  return sandboxID
    ? await CodeInterpreter.reconnect({ sandboxID, apiKey: E2B_API_KEY })
    : await CodeInterpreter.create({
        template,
        apiKey: E2B_API_KEY,
        metadata: { sessionID }
      })
}

export function nonEmpty<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined
}

export const executePythonCode: Tool[] = [
  {
    type: "function",
    function: {
      name: "execute_python_code",
      description:
        "Execute python code in Jupyter Notebook via code interpreter.",
      parameters: {
        type: "object",
        properties: {
          code: {
            type: "string",
            description: `Python code that will be directly executed via Jupyter Notebook.
                            The stdout, stderr and results will be returned as a JSON object.
                            Subsequent calls to the tool will keep the state of the interpreter.`
          }
        },
        required: ["code"]
      }
    }
  }
]
