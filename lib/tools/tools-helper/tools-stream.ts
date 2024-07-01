interface StreamProcessResult {
    fileContent: string;
    scanError: string | null;
  }
  
  export async function processStreamResponse(
    response: Response,
    sendMessage: (data: string, addExtraLineBreaks?: boolean) => void
  ): Promise<StreamProcessResult> {
    const reader = response.body?.getReader();
    const decoder = new TextDecoder();
    let fileContent = "";
    let scanError: string | null = null;
    let isFileContent = false;
    let hasOutputContent = false;
    let buffer = "";
    let isFirstOutput = true;
  
    try {
      while (true) {
        const { done, value } = await reader?.read() ?? {};
        if (done) break;
        
        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;
        const lines = buffer.split('\n');
        buffer = lines.pop() || "";
  
        for (const line of lines) {
          if (line.startsWith("SCAN_ERROR:")) {
            scanError = line.substring(11).trim();
            return { fileContent: "", scanError };
          }
  
          if (line.includes("--- FILE CONTENT START ---")) {
            isFileContent = true;
            continue;
          }
  
          if (isFileContent) {
            fileContent += line + '\n';
          } else {
            if (isFirstOutput) {
              sendMessage("**Raw output:**\n```terminal\n", false);
              isFirstOutput = false;
              hasOutputContent = true;
            }
            sendMessage(line + '\n', false);
          }
        }
      }
  
      // Process any remaining content in the buffer
      if (buffer) {
        if (isFileContent) {
          fileContent += buffer;
        } else {
          if (isFirstOutput) {
            sendMessage("**Raw output:**\n```terminal\n", false);
            hasOutputContent = true;
          }
          sendMessage(buffer, false);
        }
      }
    } finally {
      if (hasOutputContent) {
        sendMessage("```\n", false);
      }
    }
  
    return { fileContent, scanError };
  }