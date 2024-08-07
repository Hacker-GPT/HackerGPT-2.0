export interface SQLIExploiterParams {
  targetUrl: string
  method: "GET" | "POST"
  postData?: string
  enumeration: ("current-user" | "current-db" | "hostname")[]
  lightCrawling?: boolean
  cookieHeader?: string
  testParameters?: string[]
  databaseType?: string
  prefix?: string
  suffix?: string
  tamper?: string
  level: number
  risk: number
  httpCode?: number
  techniques?: string
  error: string | null
}

export const SQLI_ALLOWED_ENUM_TYPES = [
  "current-user",
  "current-db",
  "hostname"
] as const
export const SQLI_ALLOWED_TAMPERS = [
  "between",
  "charencode",
  "equaltolike",
  "space2comment",
  "base64encode"
] as const
export const SQLI_ALLOWED_TECHNIQUES = ["B", "E", "U", "S", "T", "Q"] as const

export const SQLI_DEFAULT_PARAMS: SQLIExploiterParams = {
  targetUrl: "",
  method: "GET",
  enumeration: [...SQLI_ALLOWED_ENUM_TYPES],
  level: 1,
  risk: 1,
  error: null
}

export const SQLI_FLAG_MAP: { [key: string]: keyof SQLIExploiterParams } = {
  "-u": "targetUrl",
  "-method": "method",
  "-data": "postData",
  "-enum": "enumeration",
  "-crawl": "lightCrawling",
  "-cookie": "cookieHeader",
  "-p": "testParameters",
  "-dbms": "databaseType",
  "-prefix": "prefix",
  "-suffix": "suffix",
  "-tamper": "tamper",
  "-level": "level",
  "-risk": "risk",
  "-code": "httpCode",
  "-technique": "techniques"
}

export const SQLI_MAX_INPUT_LENGTH = 1000

export type WhoisLookupParams = {
  target: string
  error?: string
}

export const WHOIS_DEFAULT_PARAMS: WhoisLookupParams = {
  target: ""
}

export const WHOIS_MAX_INPUT_LENGTH = 500

export type WafDetectorParams = {
  target: string
  error?: string
}

export const WAF_DETECTOR_DEFAULT_PARAMS: WafDetectorParams = {
  target: ""
}

export const WAF_DETECTOR_MAX_INPUT_LENGTH = 500

export interface DNSScannerParams {
  target: string
  zoneTransfer: boolean
  error?: string
}

export const DNS_SCANNER_DEFAULT_PARAMS: DNSScannerParams = {
  target: "",
  zoneTransfer: false
}

export const DNS_SCANNER_MAX_INPUT_LENGTH = 500
