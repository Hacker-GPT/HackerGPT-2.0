export interface FlagDefinitions<T> {
  [key: string]: keyof T
}

export interface CvemapParams {
  ids?: string[]
  cwes?: string[]
  vendors?: string
  products?: string
  severity?: string
  cvssScores?: string
  cpe?: string
  epssScores?: string
  epssPercentiles?: string
  age?: string
  assignees?: string
  vulnerabilityStatus?: string
  limit?: number
  output?: string
  error?: string | null
}

export const cvemapFlagDefinitions: FlagDefinitions<CvemapParams> = {
  "-id": "ids",
  "-cwe": "cwes",
  "-cwe-id": "cwes",
  "-v": "vendors",
  "-vendor": "vendors",
  "-p": "products",
  "-product": "products",
  "-s": "severity",
  "-severity": "severity",
  "-cs": "cvssScores",
  "-cvss-score": "cvssScores",
  "-c": "cpe",
  "-cpe": "cpe",
  "-es": "epssScores",
  "-epss-score": "epssScores",
  "-ep": "epssPercentiles",
  "-epss-percentile": "epssPercentiles",
  "-age": "age",
  "-a": "assignees",
  "-assignee": "assignees",
  "-vs": "vulnerabilityStatus",
  "-vstatus": "vulnerabilityStatus",
  "-l": "limit",
  "-limit": "limit",
  "-output": "output"
}

export const cvemapBooleanFlagDefinitions: FlagDefinitions<CvemapParams> = {}

export const cvemapRepeatableFlags: Set<string> = new Set([
  "-id",
  "-cwe",
  "-cwe-id"
])

export const validRcodes = [
  "noerror",
  "formerr",
  "servfail",
  "nxdomain",
  "notimp",
  "refused",
  "yxdomain",
  "yxrrset",
  "nxrrset",
  "notauth",
  "notzone",
  "badsig",
  "badvers",
  "badkey",
  "badtime",
  "badmode",
  "badname",
  "badalg",
  "badtrunc",
  "badcookie"
]

export interface DnsxParams {
  list?: string
  listFile?: string
  domain?: string[]
  domainFile?: string
  wordlist?: string[]
  wordlistFile?: string
  a?: boolean
  aaaa?: boolean
  cname?: boolean
  ns?: boolean
  txt?: boolean
  srv?: boolean
  ptr?: boolean
  mx?: boolean
  soa?: boolean
  axfr?: boolean
  caa?: boolean
  any?: boolean
  recon?: boolean
  resp?: boolean
  respOnly?: boolean
  rcode?: string[]
  cdn?: boolean
  asn?: boolean
  json?: boolean
  output?: string
  error?: string | null
}

export const dnsxFlagDefinitions: FlagDefinitions<DnsxParams> = {
  "-l": "list",
  "-list": "list",
  "-d": "domain",
  "-domain": "domain",
  "-w": "wordlist",
  "-wordlist": "wordlist",
  "-rc": "rcode",
  "-rcode": "rcode",
  "-output": "output"
}

export const dnsxBooleanFlagDefinitions: FlagDefinitions<DnsxParams> = {
  "-a": "a",
  "-aaaa": "aaaa",
  "-cname": "cname",
  "-ns": "ns",
  "-txt": "txt",
  "-srv": "srv",
  "-ptr": "ptr",
  "-mx": "mx",
  "-soa": "soa",
  "-axfr": "axfr",
  "-caa": "caa",
  "-recon": "recon",
  "-any": "any",
  "-re": "resp",
  "-resp": "resp",
  "-ro": "respOnly",
  "-resp-only": "respOnly",
  "-cdn": "cdn",
  "-asn": "asn",
  "-j": "json",
  "-json": "json"
}

export const dnsxRepeatableFlags: Set<string> = new Set([
  "-d",
  "-domain",
  "-w",
  "-wordlist",
  "-rc",
  "-rcode"
])
