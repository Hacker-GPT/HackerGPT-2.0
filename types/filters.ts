export type Filters = {
  sentiment: string
  model: string
  plugin: string
  rag: string
  reviewed: string
  date: string
}

export type FilterOption = {
  value: string
  label: string
  count: number | null
}

export type FilterOptions = {
  [key: string]: FilterOption[]
}
