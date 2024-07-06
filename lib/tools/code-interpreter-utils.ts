import "server-only"
import { CodeInterpreter, Result } from "@e2b/code-interpreter"
import { ChatCompletionTool } from "openai/resources/chat/completions"

const E2B_API_KEY = process.env.E2B_API_KEY

const sandboxTimeout = 10 * 60 * 1000 // 10 minutes in ms
const template = "code-interpreter-stateful"

export const COMMAND_GENERATION_PROMPT = `
## your job & context
you are a python data scientist. you are given tasks to complete and you run python code to solve them.
- the python code runs in jupyter notebook.
- every time you call \`execute_python\` tool, the python code is executed in a separate cell. it's okay to multiple calls to \`execute_python\`.
- display visualizations using matplotlib or any other visualization library directly in the notebook. don't worry about saving the visualizations to a file.
- you have access to the internet and can make api requests.
- you also have access to the filesystem and can read/write files.
- you can install any pip package (if it exists) if you need to but the usual packages for data analysis are already preinstalled.
- you can run any python code you want, everything is running in a secure sandbox environment.
- Make sure code is json encoded and fully complete code from zero.

## style guide
tool response values that have text inside "[]"  mean that a visual element got rendered in the notebook. for example:
- "[chart]" means that a chart was generated in the notebook.
- Always include clear and concise comments in your code to explain what each section or important line does.
`

export const CODE_INTERPRETER_TOOLS: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "execute_python",
      description:
        "Execute python code in a Jupyter notebook cell and returns any result, stdout, stderr, display_data, and error.",
      parameters: {
        type: "object",
        properties: {
          code: {
            type: "string",
            description: "The python code to execute in a single cell."
          }
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
  // console.log("Getting sandbox for session:", sessionID)
  const sandbox = await getSandbox(sessionID)

  try {
    // console.log("Executing code in sandbox:", code)
    const execution = await sandbox.notebook.execCell(code, {
      onStderr: error => console.error("Sandbox stderr:", error),
      onStdout: output => console.log("Sandbox stdout:", output),
      timeout: 60000 // 60 seconds
    })

    if (execution.error) {
      console.error("Execution error:", execution.error)
      throw new Error(execution.error.value)
    }
    // console.log("Execution results:", execution.results)
    return execution.results
  } catch (error) {
    console.error("Error in executeCode:", error)
    throw error
  } finally {
    // console.log("Keeping sandbox alive")
    await sandbox.keepAlive(sandboxTimeout).catch(console.error)
  }
}

const sandboxCache = new Map<string, CodeInterpreter>()

async function getSandbox(sessionID: string): Promise<CodeInterpreter> {
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
      metadata: { sessionID }
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
