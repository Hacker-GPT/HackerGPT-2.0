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
