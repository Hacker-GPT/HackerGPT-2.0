import { PluginID, pluginUrls, PluginSummary } from "@/types/plugins"

export const availablePlugins: PluginSummary[] = [
  {
    id: 0,
    name: "None",
    selectorName: "No plugin selected",
    value: PluginID.NONE,
    categories: [],
    isInstalled: false,
    isPremium: false,
    createdAt: "2023-01-01",
    starters: [
      {
        title: "Explain How To",
        description: "identify and exploit XSS vulnerabilities",
        chatMessage: "Explain how to identify and exploit XSS vulnerabilities."
      },
      {
        title: "Explain How To",
        description: "identify information disclosure vulnerabilities",
        chatMessage:
          "Explain how to identify information disclosure vulnerabilities."
      },
      {
        title: "Provide General Methodology",
        description: "for file upload vulnerabilities",
        chatMessage:
          "Provide General Methodology for file upload vulnerabilities."
      },
      {
        title: "Provide Techniques",
        description: "to bypass rate limit",
        chatMessage: "Provide techniques to bypass rate limit."
      }
    ]
  },
  {
    id: 1,
    name: "Auto Plugin Selector",
    selectorName: "Auto Plugin Selector",
    value: PluginID.AUTO_PLUGIN_SELECTOR,
    categories: [],
    description: "Selects the optimal plugin for your specific needs",
    icon: "https://avatars.githubusercontent.com/u/148977464",
    githubRepoUrl: pluginUrls.HACKERGPT,
    isInstalled: false,
    isPremium: false,
    createdAt: "2024-05-23",
    starters: [
      {
        title: "Start Subdomain Discovery",
        description: "for intigriti.com",
        chatMessage: "Start subdomain discovery for intigriti.com"
      },
      {
        title: "Provide Me With",
        description: "the CVEs for Microsoft that have nuclei templates",
        chatMessage:
          "Provide me with the CVEs for Microsoft that have nuclei templates."
      },
      {
        title: "Extract URLs",
        description: "from https://www.hackerone.com/product/...",
        chatMessage:
          "Extract URLs from https://www.hackerone.com/product/overview"
      },
      {
        title: "Provide Information About",
        description: "CVE-2024-23897 (critical LFI in Jenkins)",
        chatMessage:
          "Provide information about CVE-2024-23897 (critical LFI in Jenkins)."
      }
    ]
  },
  {
    id: 2,
    name: "Web Search",
    selectorName: "Web Search",
    value: PluginID.WEB_SEARCH,
    categories: ["Free"],
    icon: "https://cdn-icons-png.flaticon.com/128/364/364089.png",
    description:
      "Enhance responses with up-to-date web information and broader knowledge",
    githubRepoUrl: pluginUrls.HACKERGPT,
    isInstalled: false,
    isPremium: false,
    createdAt: "2024-06-12",
    starters: [
      {
        title: "Tell Me The Latest",
        description: "news from Hacker News",
        chatMessage: "Tell me the latest news from Hacker News."
      },
      {
        title: "Tell Me The Latest",
        description: "trends in the bug bounty",
        chatMessage: "Tell me the latest trends in the bug bounty."
      }
    ]
  },
  {
    id: 3,
    name: "CVEMap",
    selectorName: "CVEMap",
    value: PluginID.CVEMAP,
    categories: ["Free"],
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description: "Navigate the CVE jungle with ease",
    githubRepoUrl: pluginUrls.CVEMAP,
    isInstalled: false,
    isPremium: false,
    createdAt: "2024-03-13",
    starters: [
      {
        title: "Provide Me With",
        description: "the latest CVEs with the severity of critical",
        chatMessage:
          "Provide me with the latest CVEs with the severity of critical."
      },
      {
        title: "Provide Me With",
        description: "the CVEs for Microsoft that have nuclei templates",
        chatMessage:
          "Provide me with the CVEs for Microsoft that have nuclei templates."
      },
      {
        title: "Provide Information About",
        description: "CVE-2024-23897 (critical LFI in Jenkins)",
        chatMessage:
          "Provide information about CVE-2024-23897 (critical LFI in Jenkins)."
      },
      {
        title: "CVEMap Help",
        description: "How does the CVEMap plugin work?",
        chatMessage: "/cvemap -help"
      }
    ]
  },
  {
    id: 4,
    name: "Subdomain Finder",
    selectorName: "Subdomain Finder",
    categories: ["Free"],
    value: PluginID.SUBFINDER,
    icon: "https://cdn-icons-png.flaticon.com/128/3138/3138297.png",
    invertInDarkMode: true,
    description: "Discover subdomains of a domain",
    githubRepoUrl: pluginUrls.SUBFINDER,
    isInstalled: false,
    isPremium: false,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Start Subdomain Discovery",
        description: "for bugcrowd.com",
        chatMessage: "Start subdomain discovery for bugcrowd.com"
      },
      {
        title: "Scan For Active-Only",
        description: "subdomains of hackthebox.com",
        chatMessage: "Scan for active-only subdomains of hackthebox.com"
      },
      {
        title: "Scan For Subdomains",
        description: "of intigriti.com including their host IPs",
        chatMessage:
          "Scan for subdomains of intigriti.com including their host IPs."
      },
      {
        title: "Subfinder Help",
        description: "How does the Subfinder plugin work?",
        chatMessage: "/subfinder -help"
      }
    ]
  },
  {
    id: 5,
    name: "Endpoint Extractor",
    selectorName: "Endpoint Extractor",
    value: PluginID.GOLINKFINDER,
    categories: ["Free"],
    icon: "https://cdn-icons-png.flaticon.com/512/5972/5972097.png",
    description: "Fast and minimal JS endpoint extractor",
    githubRepoUrl: pluginUrls.GOLINKFINDER,
    isInstalled: false,
    isPremium: false,
    createdAt: "2024-03-26",
    starters: [
      {
        title: "Extract URLs",
        description: "from https://www.hackerone.com/product/...",
        chatMessage:
          "Extract URLs from https://www.hackerone.com/product/overview"
      },
      {
        title: "GoLinkFinder Help",
        description: "How does the GoLinkFinder plugin work?",
        chatMessage: "/golinkfinder -help"
      }
    ]
  },
  {
    id: 6,
    name: "Web Scraper",
    selectorName: "Web Scraper",
    value: PluginID.WEB_SCRAPER,
    categories: [],
    icon: "https://cdn-icons-png.flaticon.com/512/11892/11892629.png",
    description: "Extract data from websites and chat with the extracted data",
    githubRepoUrl: pluginUrls.HACKERGPT,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-03-14",
    starters: [
      {
        title: "Scrape Data",
        description: "from https://github.com/Hacker-GPT/...",
        chatMessage: "https://github.com/Hacker-GPT/HackerGPT-2.0"
      },
      {
        title: "Web Scraper Help",
        description: "How does the Web Scraper plugin work?",
        chatMessage: "How does the Web Scraper plugin work?"
      }
    ]
  },
  {
    id: 7,
    name: "Nuclei",
    selectorName: "Nuclei",
    value: PluginID.NUCLEI,
    categories: [],
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description: "Fast and customisable vulnerability scanner",
    githubRepoUrl: pluginUrls.NUCLEI,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Start Vulnerability Scan",
        description: "for hackerone.com with a focus on cves and osint",
        chatMessage:
          "Start vulnerability scan for hackerone.com with a focus on cves and osint."
      },
      {
        title: "Initiate Web Tech Detection Scan",
        description: "on hackerone.com",
        chatMessage: "/nuclei -u hackerone.com -tags tech"
      },
      {
        title: "Perform Automatic Scan",
        description: "for hackerone.com",
        chatMessage: "/nuclei -target hackerone.com -automatic-scan"
      },
      {
        title: "Nuclei Help",
        description: "How does the Nuclei plugin work?",
        chatMessage: "/nuclei -help"
      }
    ]
  },
  {
    id: 8,
    name: "Katana",
    selectorName: "Katana",
    value: PluginID.KATANA,
    categories: [],
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description:
      "Web crawling framework designed to navigate and parse for hidden details",
    githubRepoUrl: pluginUrls.KATANA,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Crawl With JavaScript Parsing",
        description: "for dynamic content on hackerone.com",
        chatMessage: "/katana -u hackerone.com -js-crawl"
      },
      {
        title: "Perform Scope-Defined Crawling",
        description: "on hackerone.com",
        chatMessage: "/katana -u hackerone.com -crawl-scope '.*hackerone.com*'"
      },
      {
        title: "Filter Content by Extension",
        description: "on target.com, excluding CSS and PNG",
        chatMessage: "/katana -u hackerone.com -extension-filter png,css"
      },
      {
        title: "Katana Help",
        description: "How does the Katana plugin work?",
        chatMessage: "/katana -help"
      }
    ]
  },
  {
    id: 9,
    name: "HTTPX",
    selectorName: "HTTPX",
    value: PluginID.HTTPX,
    categories: [],
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description:
      "Fast and multi-purpose HTTP toolkit that allows running multiple probes",
    githubRepoUrl: pluginUrls.HTTPX,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Start HTTP Analysis",
        description: "on hackerone.com, revealing server details ...",
        chatMessage: "httpx -u hackerone.com"
      },
      {
        title: "Detect Web Technologies",
        description: "on bugcrowd.com, utilizing Wappalyzer dataset...",
        chatMessage: "httpx -u bugcrowd.com -tech-detect"
      },
      {
        title: "Security Headers Analysis",
        description: "on intigriti.com, inspecting for security-...",
        chatMessage: "httpx -u intigriti.com -include-response-header -json"
      },
      {
        title: "HTTPX Help",
        description: "How does the HTTPX plugin work?",
        chatMessage: "/httpx -help"
      }
    ]
  },
  {
    id: 10,
    name: "Port Scanner",
    selectorName: "Port Scanner",
    value: PluginID.PORTSCANNER,
    categories: [],
    icon: "https://cdn-icons-png.flaticon.com/128/7338/7338907.png",
    invertInDarkMode: true,
    description: "Detect open ports and fingerprint services",
    githubRepoUrl: pluginUrls.PORTSCANNER,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-06-29",
    starters: [
      {
        title: "Perform Light Port Scan",
        description: "on hackerone.com (top 100 ports)",
        chatMessage: "Perform a light port scan on hackerone.com"
      },
      {
        title: "Scan Specific Ports",
        description: "80, 443, 8080 on hackerone.com and subdomains",
        chatMessage:
          "Scan ports 80, 443, and 8080 on hackerone.com and its subdomains: api.hackerone.com, docs.hackerone.com, resources.hackerone.com, gslink.hackerone.com"
      },
      {
        title: "Conduct Deep Port Scan",
        description: "on hackerone.com (top 1000 ports)",
        chatMessage: "Conduct a deep port scan on hackerone.com"
      },
      {
        title: "Port Scanner Help",
        description: "Display usage instructions and available options",
        chatMessage: "/portscanner -help"
      }
    ]
  },
  {
    id: 11,
    name: "GAU",
    selectorName: "GAU",
    value: PluginID.GAU,
    categories: ["Free"],
    icon: "https://avatars.githubusercontent.com/u/19563282",
    description:
      "Fetch known URLs from AlienVault's Open Threat Exchange, the Wayback Machine, and Common Crawl",
    githubRepoUrl: pluginUrls.GAU,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Start URL Enumeration",
        description: "for hackerone.com",
        chatMessage: "Enumerate URLs for hackerone.com"
      },
      {
        title: "Enumerate URLs with Date Range",
        description: "for bugcrowd.com, fetching from January to ...",
        chatMessage: "/gau bugcrowd.com --from 202301 --to 202306"
      },
      {
        title: "Enumerate URLs Including Subdomains",
        description: "for intigriti.com, capturing URLs across ...",
        chatMessage: "/gau intigriti.com --subs"
      },
      {
        title: "GAU Help",
        description: "How does the GAU plugin work?",
        chatMessage: "/gau -help"
      }
    ]
  },
  {
    id: 12,
    name: "DNS toolkit",
    selectorName: "DNS toolkit",
    value: PluginID.DNSX,
    categories: [],
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description:
      "Fast and multi-purpose DNS toolkit allow to run multiple DNS queries of your choice with a list of user-supplied resolvers",
    githubRepoUrl: pluginUrls.DNSX,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-04-19",
    starters: [
      {
        title: "Bruteforce Subdomains",
        description: "for facebook using words blog,api,beta",
        chatMessage:
          "Bruteforce subdomains for facebook using words blog, api, beta"
      },
      {
        title: "dnsX Help",
        description: "How does the dnsX plugin work?",
        chatMessage: "/dnsx -help"
      }
    ]
  },
  {
    id: 13,
    name: "AlterX",
    selectorName: "AlterX",
    categories: ["Free"],
    value: PluginID.ALTERX,
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description: "Fast and customizable subdomain wordlist generator",
    githubRepoUrl: pluginUrls.ALTERX,
    isInstalled: false,
    isPremium: false,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Generate Subdomain Wordlist",
        description: "for hackerone.com",
        chatMessage: "Generate subdomain wordlist for hackerone.com"
      },
      {
        title: "Map Subdomains Covering",
        description: "hackerone.com and its related subdomains: ...",
        chatMessage:
          "Map subdomains covering hackerone.com and its related subdomains: hackerone.com, api.hackerone.com, docs.hackerone.com, resources.hackerone.com, gslink.hackerone.com"
      },
      {
        title: "Generate Custom Enriched Wordlist",
        description: "for hackerone.com, enriching with '{{word}}-...",
        chatMessage:
          "/alterx -enrich -p '{{word}}-{{suffix}}' -list hackerone.com"
      },
      {
        title: "AlterX Help",
        description: "How does the AlterX plugin work?",
        chatMessage: "/alterx -help"
      }
    ]
  },
  {
    id: 99,
    name: "Plugins Store",
    selectorName: "Plugins Store",
    categories: [],
    value: PluginID.PLUGINS_STORE,
    isInstalled: false,
    isPremium: false,
    createdAt: "2023-01-01",
    starters: []
  }
]

export const generalPlugins = [
  {
    name: "none",
    priority: "Highest",
    description:
      "Used when no specific plugin matches the user's request, including trivia or general knowledge questions and technical inquiries not related to coding.",
    usageScenarios: [
      "If the user requests a plugin that is not available or relevant",
      "For trivia or general knowledge questions",
      "For technical questions not specifically about coding or requiring code generation or manipulation"
    ]
  },
  {
    name: "codellm",
    priority: "Medium",
    description:
      "Specifically assists in generating, updating, or debugging code. It is meant for tasks that require direct code manipulation, generation, or analysis of programming languages.",
    usageScenarios: [
      "Generate complete code scripts in various programming languages",
      "Update or refactor existing code modules",
      "Debug programs or code blocks",
      "Provide code examples and solutions for programming tasks",
      "Analyze code for best practices and potential improvements"
    ]
  },
  {
    name: "websearch",
    priority: "Medium",
    description:
      "WebSearch provides information retrieval from a pre-indexed web dataset. The information comes from periodically crawled web pages, which can be days to months old depending on indexing frequency so use this plugin when needed real-time information only.",
    usageScenarios: [
      "Find general information about recent cybersecurity trends.",
      "Look up background on AI development and its impact.",
      "Research established tech industry innovations.",
      "Gather historical data on bug bounty programs.",
      "Access broad overviews of topics that don't require absolutely current information."
    ]
  },
  {
    name: "baseLLM",
    priority: "High",
    description:
      "Handles general tasks that do not specifically match the criteria of other plugins. It leverages the base language model capabilities for a wide range of queries.",
    usageScenarios: [
      "Answer general knowledge questions not covered by other plugins.",
      "Provide explanations or summaries on various topics.",
      "Assist with tasks that require natural language understanding and generation.",
      "Handle miscellaneous queries that do not fit into specific categories."
    ]
  }
]

export function getFilteredPlugins(model: string) {
  if (model === "gpt-4-turbo-preview") {
    return generalPlugins.filter(plugin => plugin.name !== "codellm")
  }
  return generalPlugins
}

export const allFreePlugins = [
  {
    name: "none",
    priority: "Highest",
    description:
      "Used when no specific plugin matches the user's request, including trivia or general knowledge questions and technical inquiries not related to coding.",
    usageScenarios: [
      "If the user requests a plugin that is not available or relevant",
      "For trivia or general knowledge questions",
      "For technical questions not specifically about coding or requiring code generation or manipulation"
    ]
  },
  {
    name: "baseLLM",
    priority: "High",
    description:
      "Handles general tasks that do not specifically match the criteria of other plugins. It leverages the base language model capabilities for a wide range of queries.",
    usageScenarios: [
      "Answer general knowledge questions not covered by other plugins.",
      "Provide explanations or summaries on various topics.",
      "Assist with tasks that require natural language understanding and generation.",
      "Handle miscellaneous queries that do not fit into specific categories."
    ]
  },
  {
    name: "cvemap",
    priority: "Medium",
    description:
      "CVEMAP helps explore and filter CVEs database based on criteria like vendor, product/library, nuclei templates and severity.",
    usageScenarios: [
      "Get updated CVE information for a specific vendor, product, or nuclei template.",
      "Identifying vulnerabilities in specific software or libraries.",
      "Filtering CVEs by severity for risk assessment.",
      "List CVEs in specific software or libraries.",
      "Provide me with the latest CVEs with the severity of critical.",
      "Provide me with the CVEs for Microsoft that have nuclei templates."
    ]
  },
  {
    name: "websearch",
    priority: "Medium",
    description:
      "WebSearch provides information retrieval from a pre-indexed web dataset. The information comes from periodically crawled web pages, which can be days to months old depending on indexing frequency so use this plugin when needed real-time information only.",
    usageScenarios: [
      "Find general information about recent cybersecurity trends.",
      "Look up background on AI development and its impact.",
      "Research established tech industry innovations.",
      "Gather historical data on bug bounty programs.",
      "Access broad overviews of topics that don't require absolutely current information."
    ]
  },
  {
    name: "subfinder",
    priority: "Medium",
    description:
      "Subfinder discovers valid subdomains for websites using passive sources. It's fast and efficient.",
    usageScenarios: [
      "Enumerating subdomains for security testing.",
      "Gathering subdomains for attack surface analysis."
    ]
  },
  {
    name: "golinkfinder",
    priority: "Low",
    description:
      "GoLinkFinder extracts endpoints from HTML and JavaScript files, helping identify URLs within a target domain.",
    usageScenarios: [
      "Finding hidden API endpoints.",
      "Extracting URLs from web applications."
    ]
  },
  {
    name: "alterx",
    priority: "Low",
    description:
      "AlterX generates custom subdomain wordlists using DSL patterns, enriching enumeration pipelines.",
    usageScenarios: [
      "Creating wordlists for subdomain enumeration.",
      "Generating custom permutations for subdomains.",
      "Generate subdomain wordlist for a domain."
    ]
  }
]

export const allProPlugins = [
  {
    name: "nuclei",
    priority: "Medium",
    description:
      "Nuclei scans for vulnerabilities in apps, infrastructure, and networks to identify and mitigate risks.",
    usageScenarios: [
      "Scanning web applications for known vulnerabilities.",
      "Automating vulnerability assessments.",
      "Performing a comprehensive security scan on example.com",
      "Identifying outdated software versions on a target website.",
      "Running a scan to detect misconfigurations in network services.",
      "Conducting a security audit of a newly deployed application."
    ]
  },
  {
    name: "katana",
    priority: "Low",
    description:
      "Katana is a fast web crawler designed to efficiently discover endpoints in both headless and non-headless modes.",
    usageScenarios: [
      "Crawling websites to map all endpoints.",
      "Discovering hidden resources on a website."
    ]
  },
  {
    name: "httpx",
    priority: "Medium",
    description:
      "HTTPX probes web servers, gathering information like status codes, headers, and technologies.",
    usageScenarios: [
      "Analyzing server responses.",
      "Detecting technologies and services used on a server."
    ]
  },
  {
    name: "portscanner",
    priority: "Medium",
    description:
      "Port Scanner is a port scanning tool that quickly enumerates open ports on target hosts, supporting SYN, CONNECT, and UDP scans.",
    usageScenarios: [
      "Scanning for open ports on a network.",
      "Identifying accessible services on a host."
    ]
  }
  // {
  //   name: "dnsx",
  //   priority: "Low",
  //   description:
  //     "DNSX runs multiple DNS queries to discover records and perform DNS brute-forcing with user-supplied resolvers.",
  //   usageScenarios: [
  //     "Querying DNS records for a domain.",
  //     "Brute-forcing subdomains."
  //   ]
  // },
]
