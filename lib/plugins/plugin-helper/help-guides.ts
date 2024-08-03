import {
  SQLI_ALLOWED_TAMPERS,
  SQLI_ALLOWED_TECHNIQUES
} from "@/lib/tools/tool-helper/tools-flags"
import { pluginUrls } from "@/types/plugins"
import endent from "endent"

export const displayHelpGuideForCvemap = () => {
  return endent`
     [CVEMap](${pluginUrls.CVEMAP}) is an open-source command-line interface (CLI) tool that allows you to explore Common Vulnerabilities and Exposures (CVEs).
   
     ## Interaction Methods
   
     **Conversational AI Requests:**
     Engage conversationally by describing your CVE search needs in plain language. The AI will interpret your request and seamlessly execute the relevant command using CVEMap, making it user-friendly for those who prefer intuitive interactions.
     
     **Direct Commands:**
     Use direct commands by starting with "/" followed by the command and its specific flags. This method provides exact control, enabling detailed and targeted searches within the CVE database.
     
     \`\`\`
       Usage:
          /cvemap [flags]
     
       Flags:
       OPTIONS:
           -id string[]                    cve to list for given id
           -cwe, -cwe-id string[]          cve to list for given cwe id
           -v, -vendor string[]            cve to list for given vendor
           -p, -product string[]           cve to list for given product
           -s, -severity string[]          cve to list for given severity
           -cs, -cvss-score string[]       cve to list for given cvss score
           -c, -cpe string                 cve to list for given cpe
           -es, -epss-score string         cve to list for given epss score
           -ep, -epss-percentile string[]  cve to list for given epss percentile
           -age string                     cve to list published by given age in days
           -a, -assignee string[]          cve to list for given publisher assignee
           -vs, -vstatus value             cve to list for given vulnerability status in cli output. supported: new, confirmed, unconfirmed, modified, rejected, unknown
   
       OUTPUT:
           -l, -limit int       limit the number of results to display (default 25)
           -output string       write output to a file
     \`\`\`
 
     ## Examples:
 
     1. Search for critical severity CVEs related to Microsoft Windows 10:
     \`\`\`
     /cvemap -v 'microsoft' -p 'windows 10' -s critical -limit 10
     \`\`\`
 
     2. Find recent high-severity CVEs with a CVSS score greater than 8:
     \`\`\`
     /cvemap -s high -cs '> 8' -age '< 30' -limit 15
     \`\`\`
 
     3. Look for confirmed CVEs associated with a specific CWE:
     \`\`\`
     /cvemap -cwe-id 'CWE-79' -vs confirmed -limit 20
     \`\`\`

     These examples demonstrate various ways to use CVEMap for targeted CVE searches. Adjust the flags and values according to your specific needs.
     `
}

export const displayHelpGuideForKatana = () => {
  return endent`
    [Katana](${pluginUrls.KATANA}) is a fast crawler focused on execution in automation pipelines offering both headless and non-headless crawling.
    ## Interaction Methods
  
    **Conversational AI Requests:**
    Interact with Katana conversationally by simply describing your web crawling needs in plain language. The AI will understand your requirements and automatically configure and execute the appropriate Katana command, facilitating an intuitive user experience.
      
    **Direct Commands:**
    Use direct commands to specifically control the crawling process. Begin your command with the program name followed by relevant flags to precisely define the crawling scope and parameters.

    \`\`\`
      Usage:
         /katana [flags]
    
      Flags:
      INPUT:
         -u, -list string[]  target url / list to crawl
  
      CONFIGURATION:
         -d, -depth int               maximum depth to crawl (default 3)
         -jc, -js-crawl               enable endpoint parsing / crawling in javascript file
         -timeout int                 time to wait for request in seconds (default 15)
         -iqp, -ignore-query-params   Ignore crawling same path with different query-param values
  
      HEADLESS:
         -xhr, -xhr-extraction   extract xhr request url,method in jsonl output
      
      PASSIVE:
         -ps, -passive                   enable passive sources to discover target endpoints
         -pss, -passive-source string[]  passive source to use for url discovery (waybackarchive,commoncrawl,alienvault)

      SCOPE:
         -cs, -crawl-scope string[]        in scope url regex to be followed by crawler
         -cos, -crawl-out-scope string[]   out of scope url regex to be excluded by crawler
         -do, -display-out-scope           display external endpoint from scoped crawling
  
      FILTER:
         -mr, -match-regex string[]        regex or list of regex to match on output url (cli, file)
         -fr, -filter-regex string[]       regex or list of regex to filter on output url (cli, file)
         -em, -extension-match string[]    match output for given extension (eg, -em php,html,js)
         -ef, -extension-filter string[]   filter output for given extension (eg, -ef png,css)
         -mdc, -match-condition string     match response with dsl based condition
         -fdc, -filter-condition string    filter response with dsl based condition
      
      OUTPUT:
         -output string  write output to a file
    \`\`\``
}

export const displayHelpGuideForHttpx = () => {
  return endent`
    [HTTPX](${pluginUrls.HTTPX}) is a fast and multi-purpose HTTP toolkit built to support running multiple probes using a public library. Probes are specific tests or checks to gather information about web servers, URLs, or other HTTP elements. HTTPX is designed to maintain result reliability with an increased number of threads. 
  
    ## Interaction Methods
  
    **Conversational AI Requests:**
    Engage with HttpX by describing your web server analysis needs in plain language. The AI will interpret your request and automatically configure and execute the appropriate command using HTTPX, making it user-friendly for intuitive use.
    
    **Direct Commands:**
    Use direct commands to exert granular control over the probing process. Start your command with "/" followed by the necessary flags to specifically tailor your HTTP investigations.
    
    \`\`\`
      Usage:
         /httpx [flags]
    
      Flags:
      INPUT:
         -u, -target string[]  input target host(s) to probe
         -l, -list string      input file containing list of hosts to process
  
      PROBES:
         -sc, -status-code     display response status-code
         -cl, -content-length  display response content-length
         -ct, -content-type    display response content-type
         -location             display response redirect location
         -favicon              display mmh3 hash for '/favicon.ico' file
         -hash string          display response body hash (supported: md5,mmh3,simhash,sha1,sha256,sha512)
         -jarm                 display jarm fingerprint hash
         -rt, -response-time   display response time
         -lc, -line-count      display response body line count
         -wc, -word-count      display response body word count
         -title                display page title
         -bp, -body-preview    display first N characters of response body (default 100)
         -server, -web-server  display server name
         -td, -tech-detect     display technology in use based on wappalyzer dataset
         -method               display http request method
         -websocket            display server using websocket
         -ip                   display host ip
         -cname                display host cname
         -asn                  display host asn information
         -cdn                  display cdn/waf in use
         -probe                display probe status
  
      MATCHERS:
         -mc, -match-code string            match response with specified status code (-mc 200,302)
         -ml, -match-length string          match response with specified content length (-ml 100,102)
         -mlc, -match-line-count string     match response body with specified line count (-mlc 423,532)
         -mwc, -match-word-count string     match response body with specified word count (-mwc 43,55)
         -mfc, -match-favicon string[]      match response with specified favicon hash (-mfc 1494302000)
         -ms, -match-string string          match response with specified string (-ms admin)
         -mr, -match-regex string           match response with specified regex (-mr admin)
         -mcdn, -match-cdn string[]         match host with specified cdn provider (cloudfront, fastly, google, leaseweb, stackpath)
         -mrt, -match-response-time string  match response with specified response time in seconds (-mrt '< 1')
         -mdc, -match-condition string      match response with dsl expression condition
  
      EXTRACTOR:
         -er, -extract-regex string[]   display response content with matched regex
         -ep, -extract-preset string[]  display response content matched by a pre-defined regex (ipv4,mail,url)
  
      FILTERS:
         -fc, -filter-code string            filter response with specified status code (-fc 403,401)
         -fep, -filter-error-page            filter response with ML based error page detection
         -fl, -filter-length string          filter response with specified content length (-fl 23,33)
         -flc, -filter-line-count string     filter response body with specified line count (-flc 423,532)
         -fwc, -filter-word-count string     filter response body with specified word count (-fwc 423,532)
         -ffc, -filter-favicon string[]      filter response with specified favicon hash (-ffc 1494302000)
         -fs, -filter-string string          filter response with specified string (-fs admin)
         -fe, -filter-regex string           filter response with specified regex (-fe admin)
         -fcdn, -filter-cdn string[]         filter host with specified cdn provider (cloudfront, fastly, google, leaseweb, stackpath)
         -frt, -filter-response-time string  filter response with specified response time in seconds (-frt '> 1')
         -fdc, -filter-condition string      filter response with dsl expression condition
         -strip                              strips all tags in response. supported formats: html,xml (default html)
      
      OUTPUT:
         -j, -json                         write output in JSONL(ines) format
         -output string                    write output to a file
         -irh, -include-response-header    include http response (headers) in JSON output (-json only)
         -irr, -include-response           include http request/response (headers + body) in JSON output (-json only)
         -irrb, -include-response-base64   include base64 encoded http request/response in JSON output (-json only)
         -include-chain                    include redirect http chain in JSON output (-json only)
         
      OPTIMIZATIONS:
         -timeout int   timeout in seconds (default 15)
    \`\`\``
}

export const displayHelpGuideForNuclei = () => {
  return endent`
    [Nuclei](${pluginUrls.NUCLEI}) is a fast exploitable vulnerability scanner designed to probe modern applications, infrastructure, cloud platforms, and networks, aiding in the identification and mitigation of vulnerabilities. 
  
    ## Interaction Methods
  
    **Conversational AI Requests:**
    Engage with Nuclei by describing your vulnerability scanning needs in plain language. The AI will interpret your request and automatically configure and execute the appropriate command using Nuclei, simplifying the user experience.
    
    **Direct Commands:**
    Utilize direct commands to precisely control the scanning process. Begin your command with "/" followed by the necessary flags to tailor your scans to specific targets and conditions.
    
    \`\`\`
      Usage:
         /nuclei [flags]
    
      Flags:
      TARGET:
         -u, -target string[]          target URLs/hosts to scan
         -eh, -exclude-hosts string[]  hosts to exclude to scan from the input list (ip, cidr, hostname)
         -sa, -scan-all-ips            scan all the IP's associated with dns record
         -iv, -ip-version string[]     IP version to scan of hostname (4,6) - (default 4)
  
      TEMPLATES:
         -nt, -new-templates                    run only new templates added in latest nuclei-templates release
         -ntv, -new-templates-version string[]  run new templates added in specific version
         -as, -automatic-scan                   automatic web scan using wappalyzer technology detection to tags mapping
         -t, -templates string[]                list of template to run (comma-separated)
         -turl, -template-url string[]          template url to run (comma-separated)
         -w, -workflows string[]                list of workflow to run (comma-separated)
         -wurl, -workflow-url string[]          workflow url to run (comma-separated)
         -code                                  enable loading code protocol-based templates
        
      FILTERING:
         -a, -author string[]               templates to run based on authors (comma-separated)
         -tags string[]                     templates to run based on tags (comma-separated) Possible values: cves, osint, tech ...)
         -etags, -exclude-tags string[]     templates to exclude based on tags (comma-separated)
         -itags, -include-tags string[]     tags to be executed even if they are excluded either by default or configuration
         -id, -template-id string[]         templates to run based on template ids (comma-separated, allow-wildcard)
         -eid, -exclude-id string[]         templates to exclude based on template ids (comma-separated)
         -it, -include-templates string[]   templates to be executed even if they are excluded either by default or configuration
         -et, -exclude-templates string[]   template or template directory to exclude (comma-separated)
         -em, -exclude-matchers string[]    template matchers to exclude in result
         -s, -severity value[]              templates to run based on severity. Possible values: info, low, medium, high, critical, unknown
         -es, -exclude-severity value[]     templates to exclude based on severity. Possible values: info, low, medium, high, critical, unknown
         -pt, -type value[]                 templates to run based on protocol type. Possible values: dns, file, http, headless, tcp, workflow, ssl, websocket, whois, code, javascript
         -ept, -exclude-type value[]        templates to exclude based on protocol type. Possible values: dns, file, http, headless, tcp, workflow, ssl, websocket, whois, code, javascript
         -tc, -template-condition string[]  templates to run based on expression condition
  
      OUTPUT:
         -j, -jsonl      write output in JSONL(ines) format
         -output string  write output to a file
  
      CONFIGURATIONS:
         -fr, -follow-redirects          enable following redirects for http templates
         -fhr, -follow-host-redirects    follow redirects on the same host
         -mr, -max-redirects int         max number of redirects to follow for http templates (default 10)
         -dr, -disable-redirects         disable redirects for http templates
         -H, -header string[]            custom header/cookie to include in all http request in header:value format (cli)
         -V, -var value                  custom vars in key=value format
         -sr, -system-resolvers          use system DNS resolving as error fallback
         -dc, -disable-clustering        disable clustering of requests
         -passive                        enable passive HTTP response processing mode
         -fh2, -force-http2              force http2 connection on requests
         -dt, -dialer-timeout value      timeout for network requests.
         -dka, -dialer-keep-alive value  keep-alive duration for network requests.
         -at, -attack-type string        type of payload combinations to perform (batteringram,pitchfork,clusterbomb)
  
      OPTIMIZATIONS:
         -timeout int               time to wait in seconds before timeout (default 30)
         -mhe, -max-host-error int  max errors for a host before skipping from scan (default 30)
         -nmhe, -no-mhe             disable skipping host from scan based on errors
         -ss, -scan-strategy value  strategy to use while scanning(auto/host-spray/template-spray) (default auto)
         -nh, -no-httpx             disable httpx probing for non-url input
    \`\`\``
}

export const displayHelpGuideForLinkFinder = () => {
  return endent`
    [Link Finder](${pluginUrls.LINKFINDER}) is a minimalistic JavaScript endpoint extractor that efficiently pulls endpoints from HTML and embedded JavaScript files. 
  
    ## Interaction Methods
  
    **Conversational AI Requests:**
    Interact with LinkFinder using plain language to describe your endpoint extraction needs. The AI will understand your input and automatically execute the corresponding command with LinkFinder, simplifying the process for intuitive use.
    
    **Direct Commands:**
    For precise and specific tasks, use direct commands. Begin your command with "/" and include necessary flags to control the extraction process effectively.
    
    \`\`\`
      Usage:
         /linkfinder --domain [domain]
  
      Flags:
      CONFIGURATION:
         -d --domain string   Input a URL.

      OUTPUT:  
         --output string  write output to a file
    \`\`\``
}

export const displayHelpGuideForGau = () => {
  return endent`
    [GAU](${pluginUrls.GAU}) is a powerful web scraping tool that fetches known URLs from multiple sources, including AlienVault&apos;s Open Threat Exchange, the Wayback Machine, and Common Crawl. 
  
    ## Interaction Methods
  
    **Conversational AI Requests:**
    Engage with GAU conversationally by simply describing your URL fetching needs in plain language. The AI will interpret your input and carry out the necessary operations automatically, providing an intuitive and user-friendly experience.
    
    **Direct Commands:**
    Use direct commands for meticulous control. Start with "/" followed by the command and applicable flags to perform specific URL fetching tasks tailored to your requirements.
    
    \`\`\`
      Usage:
         /gau [target] [flags]
  
      Flags:
      CONFIGURATION:
         --from string         fetch URLs from date (format: YYYYMM)
         --to string           fetch URLs to date (format: YYYYMM)
         --providers strings   list of providers to use (wayback, commoncrawl, otx, urlscan)
         --subs                include subdomains of target domain
  
      FILTER:
         --blacklist strings   list of extensions to skip
         --fc strings          list of status codes to filter
         --ft strings          list of mime-types to filter
         --mc strings          list of status codes to match
         --mt strings          list of mime-types to match
         --fp                  remove different parameters of the same endpoint

      OUTPUT:  
         --output string  write output to a file
    \`\`\``
}

// TOOLS
export const displayHelpGuideForPortScanner = () => {
  return endent`
     [Port Scanner](${pluginUrls.PORTSCANNER}) is a fast and efficient tool for scanning ports on specified hosts.
 
     ## Interaction Methods
 
     **Conversational AI Requests:**
     Engage with Port Scanner by describing your port scanning needs in plain language. The AI will interpret your request and automatically execute the relevant command, offering a user-friendly interface for those who prefer intuitive interactions.
 
     **Direct Commands:**
     Utilize direct commands for granular control over port scanning. Start your command with "/" followed by the necessary flags to specify detailed parameters for the scan.
     
     \`\`\`
     Usage:
        /portscanner [flags]
      
     Flags:
     INPUT:
        -host string[]          hosts to scan ports for (comma-separated)
        -scan-type, -st string  type of scan to perform: "light", "deep", or "custom"
        -port, -p               specific ports to scan [80,443, 100-200] (for custom scan type) 
        -top-ports, -tp string  number of top ports to scan [full,100,1000] (for custom scan type)
        -no-svc                 disable service detection
 
     Scan Types:
        light:   scans the top 100 most common ports
        deep:    scans the top 1000 most common ports
        custom:  allows you to specify ports or top ports to scan
 
     Examples:
        /portscanner -host example.com -scan-type light
        /portscanner -host 104.18.36.214,23.227.38.33 -scan-type deep
        /portscanner -host example.com -scan-type custom -port 80,443,8080
        /portscanner -host example.com -scan-type custom -top-ports 500
      \`\`\``
}

export const displayHelpGuideForSubdomainFinder = () => {
  return endent`
    [Subdomain Finder](${pluginUrls.SUBFINDER}) is a powerful subdomain discovery tool designed to enumerate and uncover valid subdomains of websites efficiently through passive online sources. 
  
    ## Interaction Methods
  
    **Conversational AI Requests:**
    Engage with Subfinder by describing your subdomain discovery needs in plain language. The AI will interpret your request and automatically execute the relevant command with Subfinder, offering a user-friendly interface for those who prefer intuitive interactions.
    
    **Direct Commands:**
    Utilize direct commands for granular control over subdomain discovery. Start your command with "/" followed by the necessary flags to specify detailed parameters for the scan.
    
    \`\`\`
    Usage:
    /subfinder [flags]
 
   Flags:
   INPUT:
      -d, -domain string[]   domains to find subdomains for (comma-separated)

   CONFIGURATION:
      -nW, -active           display active subdomains only

   OUTPUT:
      -oJ, -json             write output in JSONL(ines) format
      -output string         write output to a file
      -oI, -ip               include host IP in output (-active only)
   
   Examples:
      /subfinder -d example.com
      /subfinder -d example.com -output subdomains.txt
      /subfinder -d example.com,example.org -active
      /subfinder -d example.com -json -active -ip
    \`\`\``
}

export const displayHelpGuideForSSLScanner = () => {
  return endent`
      [SSL/TLS Scanner](${pluginUrls.SSLSCANNER}) is a specialized tool for analyzing SSL/TLS configurations and vulnerabilities on specified hosts.
  
      ## Interaction Methods
  
      **Conversational AI Requests:**
      Engage with SSL Scanner by describing your SSL/TLS scanning needs in plain language. The AI will interpret your request and automatically execute the relevant command, offering a user-friendly interface for those who prefer intuitive interactions.
  
      **Direct Commands:**
      Utilize direct commands for granular control over SSL/TLS scanning. Start your command with "/" followed by the necessary flags to specify detailed parameters for the scan.
      
      \`\`\`
      Usage:
         /sslscanner [flags]
       
      Flags:
      INPUT:
         -host string            single host to scan (domain or IP address)
         -scan-type, -st string  type of scan to perform: "light", "deep", or "custom"
         -port, -p string        specific ports to scan [443,8443] (for custom scan type) 
         -top-ports, -tp string  number of top SSL/TLS ports to scan [full,100,1000] (for custom scan type)
         -no-vuln-check          disable vulnerability checks
  
      Scan Types:
         light:   scans only port 443 of the provided domain or IP (default)
         deep:    performs a thorough scan of SSL/TLS configurations and vulnerabilities on the top 1000 ports
         custom:  allows you to specify ports or top ports to scan
  
      IMPORTANT:
         - Only one target (domain or IP) can be scanned at a time.
         - Light scan (default) only checks port 443.
         - Deep scan checks the top 1000 ports.
  
      Examples:
         /sslscanner -host example.com
         /sslscanner -host 104.18.36.214 -scan-type deep
         /sslscanner -host example.com -scan-type custom -port 443,8443
         /sslscanner -host example.com -scan-type deep -no-vuln-check
       \`\`\``
}

export const displayHelpGuideForSQLIExploiter = () => {
  return endent`
   [SQLi Exploiter](${pluginUrls.SQLIEXPLOITER}) is a powerful tool for detecting and exploiting SQL injection vulnerabilities in web applications.
 
   ## Interaction Methods
 
   **Conversational AI Requests:**
   
   Engage with SQLi Exploiter by describing your SQL injection testing needs in plain language. The AI will interpret your request and automatically execute the relevant command, offering a user-friendly interface for those who prefer intuitive interactions.
 
   **Direct Commands:**

   Utilize direct commands for granular control over SQL injection testing. Start your command with "/" followed by the necessary flags to specify detailed parameters for the scan.
 
   \`\`\`
   Usage:
      /sqliexploiter [flags]
 
   Flags:
   INPUT:
      -u string             Target URL to scan (must start with http:// or https://)
      -method string        HTTP method to use: GET or POST (default: GET)
      -data string          POST data to include in the payload (e.g., "id=1")
      -enum string          Data to extract: current-user, current-db, hostname (comma-separated)
      -crawl                Enable light crawling of the target
      -cookie string        HTTP Cookie header to include in each request
      -p string             Comma-separated list of parameters to test
      -dbms string          Force specific database type for payloads
      -prefix string        String to prepend to each payload
      -suffix string        String to append to each payload
      -tamper string        Script to modify payloads (${SQLI_ALLOWED_TAMPERS.join(", ")})
      -level int            Level of tests to perform (1-5, default: 1)
      -risk int             Risk of tests to perform (1-3, default: 1)
      -code int             HTTP code to match when a query is evaluated to True
      -technique string     SQLi techniques to use (${SQLI_ALLOWED_TECHNIQUES.join("")})
      \`\`\`

   **IMPORTANT:**
      - All URLs must start with http:// or https://
      - Higher levels and risks increase scan time and potential impact on the target
      - Use advanced options carefully, especially on production systems
 
   **Examples:**
      \`\`\`
      /sqliexploiter -u http://example.com/page.php?id=1
      /sqliexploiter -u http://example.com/login.php -method POST -data "username=admin&password=test"
      /sqliexploiter -u http://example.com/search.php?q=test -enum current-user,current-db -level 3 -risk 2
      /sqliexploiter -u http://example.com/page.php?id=1 -crawl -tamper space2comment -technique BEUSTQ
      \`\`\`

   **Additional Information:**
      - Level: Higher levels test more entry points (e.g., cookies, headers) but take longer.
      - Risk: Higher risks include more aggressive tests that might impact database performance.
      - Techniques: B (Boolean), E (Error), U (UNION), S (Stacked queries), T (Time-based), Q (Inline queries)
      `
}

export const displayHelpGuideForWhoisLookup = () => {
  return endent`
     [Whois Lookup](${pluginUrls.WHOIS}) is a tool for querying domain name and IP address ownership information.
 
     ## Interaction Methods
 
     **Conversational AI Requests:**
     
     Engage with Whois Lookup by describing your domain or IP lookup needs in plain language. The AI will interpret your request and automatically execute the relevant command, offering a user-friendly interface for those who prefer intuitive interactions.
 
     **Direct Commands:**
 
     Utilize direct commands for precise Whois lookups. Start your command with "/" followed by the necessary flags to specify the target and options.
 
     \`\`\`
     Usage:
        /whois [flags]
 
     Flags:
     INPUT:
        -t, -target string   Target domain or IP address to lookup
     \`\`\`
 
     **IMPORTANT:**
        - The target can be either a domain name or an IP address
        - Some Whois servers may have rate limits, so use responsibly
 
     **Examples:**
        \`\`\`
        /whois -t example.com
        /whois -t 8.8.8.8
        \`\`\`
 
     **Additional Information:**
        - Whois provides information such as:
          * Domain registrar and registration dates
          * Name servers
          * Registrant contact information (if available)
          * IP address range and associated organization
        - Some information may be redacted for privacy reasons
        - The raw output option can provide more detailed information, but may be harder to read
     `
}
