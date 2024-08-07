import { pluginUrls } from "@/types/plugins"
import endent from "endent"

export const displayHelpGuideForWAFDetector = () => {
  return endent`
         [WAF Detector](${pluginUrls.WAFDETECTOR}) is a tool designed to fingerprint the Web Application Firewall (WAF) behind a target application.
  
         Discover if a website is protected by a web application firewall (WAF) and identify which WAF is in use. This is a crucial step in the information gathering phase of a penetration test, allowing pentesters to adapt their attacks for successful bypasses.
    
         ## Usage
         \`\`\`
         /wafdetector [flags]
     
         Flags:
         INPUT:
            -t, -target string   Target URL or domain to fingerprint WAF
         \`\`\`
     
         ## Examples
         \`\`\`
         /wafdetector -t example.com
         /wafdetector -t https://example.com
         \`\`\`

         **Interaction Methods**
         Interact with WAF Detector using natural language queries or direct commands starting with "/", followed by the necessary flags to specify the target and options.

         **Note:**
         This tool uses WAFW00F for WAF detection. WAFW00F is distributed under the BSD 3-Clause License. For full license details, please visit: https://github.com/EnableSecurity/wafw00f/blob/master/LICENSE

         **Usage Disclaimer:**
         This tool should only be used on websites you have permission to test. Unauthorized scanning may be illegal.
         `
}

export const displayHelpGuideForDNSScanner = () => {
  return endent`
    [DNS Scanner](${pluginUrls.DNSSCANNER}) is a tool designed to perform DNS reconnaissance and gather information about domain name systems.

    This tool allows you to query DNS servers for various types of records and perform zone transfer attempts. It's an essential step in the information gathering phase of a security assessment, providing valuable insights into a target's DNS infrastructure.

    ## Usage
    \`\`\`
    /dnsscanner [flags]

    Flags:
    INPUT:
      -target, -t string   Target domain to scan
      -zone-transfer, -z   Test all NS servers for a zone transfer
    \`\`\`

    ## Examples
    \`\`\`
    /dnsscanner -target example.com
    /dnsscanner -target example.com -zone-transfer
    \`\`\`

    **Interaction Methods**
    Interact with DNS Scanner using natural language queries or direct commands starting with "/", followed by the necessary flags to specify the target and options.
  `
}
