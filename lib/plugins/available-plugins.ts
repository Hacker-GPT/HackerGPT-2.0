import { PluginID, pluginUrls, PluginSummary } from "@/types/plugins"

export const availablePlugins: PluginSummary[] = [
  {
    id: 0,
    name: "None",
    selectorName: "No plugin selected",
    value: PluginID.NONE,
    categories: [],
    isInstalled: false,
    isPremium: true,
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
        description: "to bypass two-factor authentication",
        chatMessage: "Provide techniques to bypass two-factor authentication."
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
    isPremium: true,
    createdAt: "2024-05-23",
    starters: [
      {
        title: "Start Subdomain Discovery",
        description: "for reddit.com",
        chatMessage: "Start subdomain discovery for reddit.com"
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
    name: "Web Browse",
    selectorName: "Web Browse",
    value: PluginID.WEB_BROWSE,
    categories: ["Free"],
    icon: "https://cdn-icons-png.flaticon.com/512/364/364089.png",
    description:
      "HackerGPT with internet access, enabling real-time information retrieval to enhance responses",
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
        description: "trends in the web application hacking field",
        chatMessage:
          "Tell me the latest trends in the web application hacking field."
      }
    ]
  },
  {
    id: 3,
    name: "CVEMap",
    selectorName: "CVEMap: CVE Database",
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
    name: "Subfinder",
    selectorName: "Subfinder: Subdomain Finder",
    categories: [],
    value: PluginID.SUBFINDER,
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description: "Fast passive subdomain enumeration tool",
    githubRepoUrl: pluginUrls.SUBFINDER,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Start Subdomain Discovery",
        description: "for reddit.com",
        chatMessage: "Start subdomain discovery for reddit.com"
      },
      {
        title: "Scan For Active-Only",
        description: "subdomains of hackthebox.com",
        chatMessage: "Scan for active-only subdomains of hackthebox.com"
      },
      {
        title: "Scan For Subdomains",
        description: "of netflix.com including their host IPs",
        chatMessage:
          "Scan for subdomains of netflix.com including their host IPs."
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
    name: "GoLinkFinder",
    selectorName: "GoLinkFinder: Endpoint Extractor",
    value: PluginID.GOLINKFINDER,
    categories: [],
    icon: "https://cdn-icons-png.flaticon.com/512/5972/5972097.png",
    description: "Fast and minimal JS endpoint extractor",
    githubRepoUrl: pluginUrls.GOLINKFINDER,
    isInstalled: false,
    isPremium: true,
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
    selectorName: "Web Scraper: Website Data Extractor",
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
    selectorName: "Nuclei: Website Vulnerability Scanner",
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
  // {
  //   id: 7,
  //   name: "Amass",
  //   selectorName: "Amass: Network mapping",
  //   value: PluginID.AMASS,
  //   icon: "https://www.kali.org/tools/amass/images/amass-logo.svg",
  //   description: "In-depth attack surface mapping and asset discovery",
  //   githubRepoUrl: pluginUrls.AMASS,
  //   isInstalled: false,
  //   isPremium: true,
  //   starters: [
  //     {
  //       title: "Enumerate",
  //       description: "hackerone.com",
  //       chatMessage: "/amass enum -d hackerone.com"
  //     },
  //     {
  //       title: "Find ASNs",
  //       description: "for Tesla",
  //       chatMessage: "/amass intel -org 'Tesla'"
  //     },
  //     {
  //       title: "Perform Passive Enumeration",
  //       description: "for tesla.com",
  //       chatMessage: "/amass enum -passive -d tesla.com"
  //     },
  //     {
  //       title: "Amass Help",
  //       description: "How does the Amass plugin work?",
  //       chatMessage: "/amass -help"
  //     }
  //   ]
  // },
  {
    id: 8,
    name: "Katana",
    selectorName: "Katana: Web Crawling",
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
    selectorName: "HTTPX: Web Analysis",
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
        description: "on hackerone.com, utilizing Wappalyzer dataset...",
        chatMessage: "httpx -u hackerone.com -tech-detect"
      },
      {
        title: "Security Headers Analysis",
        description: "on hackerone.com, inspecting for security-...",
        chatMessage: "httpx -u hackerone.com -include-response-header -json"
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
    name: "Naabu",
    selectorName: "Naabu: Port Scanner",
    value: PluginID.NAABU,
    categories: [],
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description:
      "Fast port scanner written in go with a focus on reliability and simplicity",
    githubRepoUrl: pluginUrls.NAABU,
    isInstalled: false,
    isPremium: true,
    createdAt: "2024-02-27",
    starters: [
      {
        title: "Start Port Scanning",
        description: "for shopify.com",
        chatMessage: "Start port scanning for shopify.com"
      },
      {
        title: "Scan ports 80, 443, and 8080",
        description: "for hackerone.com and its subdomains: ...",
        chatMessage:
          "Scan ports 80, 443, and 8080 for hackerone.com and its subdomains: api.hackerone.com, docs.hackerone.com, resources.hackerone.com, gslink.hackerone.com"
      },
      {
        title: "Scan Top 1000 Ports",
        description: "on tesla.com, excluding ports 21 and 22",
        chatMessage:
          "Scan top 1000 ports on tesla.com, excluding ports 21 and 22."
      },
      {
        title: "Naabu Help",
        description: "How does the Naabu plugin work?",
        chatMessage: "/naabu -help"
      }
    ]
  },
  {
    id: 11,
    name: "GAU",
    selectorName: "GAU: Url Enumeration",
    value: PluginID.GAU,
    categories: [],
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
        description: "for tesla.com",
        chatMessage: "Enumerate URLs for tesla.com"
      },
      {
        title: "Enumerate URLs with Date Range",
        description: "for tesla.com, fetching from January to ...",
        chatMessage: "/gau tesla.com --from 202301 --to 202306"
      },
      {
        title: "Enumerate URLs Including Subdomains",
        description: "for tesla.com, capturing URLs across ...",
        chatMessage: "/gau tesla.com --subs"
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
    name: "dnsX",
    selectorName: "dnsX: DNS toolkit",
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
    selectorName: "AlterX: Subdomain Wordlist Generator",
    categories: [],
    value: PluginID.ALTERX,
    icon: "https://avatars.githubusercontent.com/u/50994705",
    description: "Fast and customizable subdomain wordlist generator",
    githubRepoUrl: pluginUrls.ALTERX,
    isInstalled: false,
    isPremium: true,
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
        description: "for tesla.com, enriching with '{{word}}-...",
        chatMessage: "/alterx -enrich -p '{{word}}-{{suffix}}' -list tesla.com"
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
