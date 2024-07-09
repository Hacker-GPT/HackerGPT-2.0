export type Filters = {
  sentiment: string
  model: string
  plugin: string
  rag: string
  reviewed: string
}

export type FilterOption = {
  value: string
  label: string
  count: number
}

export type FilterOptions = {
  [key: string]: FilterOption[]
}
