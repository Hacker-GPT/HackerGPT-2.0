import { Message } from "@/types/chat"
import endent from "endent"

export const transformUserQueryToWafdetectorLookupCommand = (
  lastMessage: Message
) => {
  const answerMessage = endent`
      Query: "${lastMessage.content}"
  
      Based on this query, generate a command for the 'wafdetector' tool, which designed to fingerprint the Web Application Firewall (WAF) behind a target application. The command should follow this structured format:
  
      ALWAYS USE THIS FORMAT:
      \`\`\`json
      { "command": "wafdetector -t [target]" }
      \`\`\`
      Replace '[target]' with the actual domain or URL of the website to be scanned. Ensure the command is properly escaped to be valid JSON.
    
      Command Construction Guidelines:
      1. **Target Specification** (Required):
        - -t, --target string: Specifies the target domain or URL for the WAF detection scan.
  
      IMPORTANT:
      - Generate only one command at a time.
      - Only one target (domain or URL) can be scanned at a time.
      - Use -help flag when the user asks for help or information about the tool.
  
      Example Commands:
      For a basic scan with domain:
      \`\`\`json
      { "command": "wafdetector -t example.com" }
      \`\`\`
  
      For a request for help:
      \`\`\`json
      { "command": "wafdetector -help" }
      \`\`\`
    
      Response:`

  return answerMessage
}
