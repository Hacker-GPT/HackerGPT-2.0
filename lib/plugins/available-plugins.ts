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
  // {
  //   id: 1,
  //   name: "Auto Plugin Selector",
  //   selectorName: "Auto Plugin Selector",
  //   value: PluginID.AUTO_PLUGIN_SELECTOR,
  //   categories: [],
  //   description: "Selects the optimal plugin for your specific needs",
  //   icon: "https://avatars.githubusercontent.com/u/148977464",
  //   githubRepoUrl: pluginUrls.HACKERGPT,
  //   isInstalled: false,
  //   isPremium: false,
  //   createdAt: "2024-05-23",
  //   starters: [
  //     {
  //       title: "Start Subdomain Discovery",
  //       description: "for intigriti.com",
  //       chatMessage: "Start subdomain discovery for intigriti.com"
  //     },
  //     {
  //       title: "Provide Me With",
  //       description: "the CVEs for Microsoft that have nuclei templates",
  //       chatMessage:
  //         "Provide me with the CVEs for Microsoft that have nuclei templates."
  //     },
  //     {
  //       title: "Extract URLs",
  //       description: "from https://www.hackerone.com/product/...",
  //       chatMessage:
  //         "Extract URLs from https://www.hackerone.com/product/overview"
  //     },
  //     {
  //       title: "Provide Information About",
  //       description: "CVE-2024-23897 (critical LFI in Jenkins)",
  //       chatMessage:
  //         "Provide information about CVE-2024-23897 (critical LFI in Jenkins)."
  //     }
  //   ]
  // },
  // {
  //   id: 2,
  //   name: "Web Search",
  //   selectorName: "Web Search",
  //   value: PluginID.WEB_SEARCH,
  //   categories: ["Free"],
  //   icon: "https://cdn-icons-png.flaticon.com/128/11193/11193901.png",
  //   invertInDarkMode: true,
  //   description:
  //     "Enhance responses with up-to-date web information and broader knowledge",
  //   githubRepoUrl: pluginUrls.HACKERGPT,
  //   isInstalled: false,
  //   isPremium: false,
  //   createdAt: "2024-06-12",
  //   starters: [
  //     {
  //       title: "Tell Me The Latest",
  //       description: "news from Hacker News",
  //       chatMessage: "Tell me the latest news from Hacker News."
  //     },
  //     {
  //       title: "Tell Me The Latest",
  //       description: "trends in the bug bounty",
  //       chatMessage: "Tell me the latest trends in the bug bounty."
  //     }
  //   ]
  // },
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
    name: "Link Finder",
    selectorName: "Link Finder",
    value: PluginID.LINKFINDER,
    categories: ["Free"],
    icon: "https://cdn-icons-png.flaticon.com/128/9465/9465808.png",
    invertInDarkMode: true,
    description: "Fast and minimal JS endpoint extractor",
    githubRepoUrl: pluginUrls.LINKFINDER,
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
        title: "Link Finder Help",
        description: "How does the Link Finder plugin work?",
        chatMessage: "/linkfinder -help"
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
    name: "Subdomain Wordlist Generator",
    selectorName: "Subdomain Wordlist Generator",
    categories: ["Free"],
    value: PluginID.ALTERX,
    icon: "https://cdn-icons-png.flaticon.com/128/1027/1027221.png",
    invertInDarkMode: true,
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
    name: "codellm",
    priority: "Medium",
    description:
      "Specialized plugin for comprehensive code-related tasks. Handles generation and modification of complete code scripts across various programming languages.",
    usageScenarios: [
      "Creating complete code scripts in multiple programming languages",
      "Modifying or refactoring existing complete code scripts",
      "Providing full code examples and solutions for specific programming challenges",
      "Analyzing and optimizing complete code scripts for best practices"
    ]
  },
  {
    name: "websearch",
    priority: "Low",
    description:
      "Information retrieval tool accessing a pre-indexed web dataset. Provides data from periodically crawled web pages, which may be days to months old. Best used for recent but not real-time information needs.",
    usageScenarios: [
      "Researching recent (but not current) cybersecurity trends",
      "Gathering information on recent developments in AI",
      "Exploring established tech industry innovations",
      "Collecting data on historical bug bounty programs",
      "Accessing comprehensive overviews of topics not requiring absolutely current information"
    ]
  },
  {
    name: "baseLLM",
    priority: "High",
    description:
      "Handles a wide range of tasks and provides comprehensive information on various fields, including cybersecurity and programming. Uses the base language model's extensive knowledge for queries that don't require real-time data or specialized tools.",
    usageScenarios: [
      "Answering general and specific knowledge questions on established topics",
      "Providing detailed explanations, methodologies, and code examples for technical concepts",
      "Assisting with tasks that require in-depth understanding of cybersecurity, programming, or other technical fields",
      "Handling queries about vulnerabilities, attack methods, or security practices, including specific code examples when appropriate",
      "Generating code snippets, explanations, and technical details for various programming and security-related tasks"
    ]
  }
]

export function getFilteredPlugins(model: string) {
  if (model === "gpt-4-turbo-preview") {
    return generalPlugins.filter(plugin => plugin.name !== "codellm")
  }
  return generalPlugins
}
