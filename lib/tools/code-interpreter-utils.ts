import "server-only"
import { CodeInterpreter, Result } from "@e2b/code-interpreter"
import { ChatCompletionTool } from "openai/resources/chat/completions"

const E2B_API_KEY = process.env.E2B_API_KEY
if (!E2B_API_KEY) throw new Error("E2B_API_KEY environment variable not found")

const sandboxTimeout = 10 * 60 * 1000 // 10 minutes in ms
const template = "code-interpreter-stateful"

export const SYSTEM_PROMPT = `
You are a Python data scientist. Run Python code in a Jupyter notebook to solve tasks.
- Display visualizations using matplotlib or other libraries.
- You can make API requests and access the filesystem.
- Install pip packages if needed; common data analysis packages are pre-installed.
- "[chart]" in tool responses means a chart was generated.
`

export const CODE_INTERPRETER_TOOLS: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "execute_python",
      description: "Execute Python code in a Jupyter notebook cell.",
      parameters: {
        type: "object",
        properties: {
          code: { type: "string", description: "Python code to execute" }
        },
        required: ["code"]
      }
    }
  }
]

export async function executeCode(
  sessionID: string,
  code: string
): Promise<Result[]> {
  const sandbox = await getSandbox(sessionID)

  try {
    const execution = await sandbox.notebook.execCell(code, {
      onStderr: console.error,
      onStdout: console.log,
      timeout: 25000
    })

    if (execution.error) throw new Error(execution.error.value)
    return execution.results
  } finally {
    try {
      await sandbox.keepAlive(sandboxTimeout)
    } catch {
      // Ignore keepalive errors
    }
    await sandbox.close()
  }
}

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
