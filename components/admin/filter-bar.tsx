import { Button } from "@/components/ui/button"
import { FilterOptions, Filters } from "@/types/filters"
import { useState } from "react"
import { ChevronDown } from "lucide-react"

type FilterBarProps = {
  filterOptions: FilterOptions
  filters: Filters
  handleFilterChange: (key: string, value: any) => void
  handleClearAllFilters: () => void
  customDateRange: { start: string; end: string }
  handleCustomDateChange: (start: string, end: string) => void
}

export function FilterBar({
  filterOptions,
  filters,
  handleFilterChange,
  handleClearAllFilters,
  customDateRange,
  handleCustomDateChange
}: FilterBarProps) {
  const [showCustomDate, setShowCustomDate] = useState(false)

  return (
    <div className="mb-8 rounded-xl bg-white p-6 shadow-md transition-shadow duration-300 hover:shadow-lg dark:bg-gray-800">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white">
          Filters
        </h2>
        <Button
          variant="outline"
          size="sm"
          onClick={handleClearAllFilters}
          className="bg-gray-100 text-xs font-medium text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
        >
          Clear All Filters
        </Button>
      </div>

      <div className="flex flex-wrap gap-6">
        {Object.entries(filterOptions).map(([key, options]) => (
          <div key={key} className="flex flex-col">
            <span className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-600 dark:text-gray-300">
              {key === "date" ? "Time Period" : key}
            </span>
            {key === "date" ? (
              <DateFilter
                options={options}
                value={filters.date}
                onChange={(value: any) => {
                  handleFilterChange(key, value)
                  setShowCustomDate(value === "custom")
                }}
                showCustomDate={showCustomDate}
                customDateRange={customDateRange}
                handleCustomDateChange={handleCustomDateChange}
              />
            ) : (
              <ButtonGroup
                options={options}
                selectedValue={filters[key as keyof Filters]}
                onChange={(value: any) => handleFilterChange(key, value)}
              />
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

function DateFilter({
  options,
  value,
  onChange,
  showCustomDate,
  customDateRange,
  handleCustomDateChange
}: any) {
  return (
    <div className="flex gap-2">
      <div className="relative">
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          className="appearance-none rounded-md border border-gray-300 bg-white px-3 py-1.5 pr-8 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
        >
          {options.map((option: any) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-2 top-1/2 size-4 -translate-y-1/2 text-gray-400" />
      </div>
      {showCustomDate && (
        <div className="flex gap-2">
          <DateInput
            value={customDateRange.start}
            onChange={(newDate: any) =>
              handleCustomDateChange(newDate, customDateRange.end)
            }
          />
          <DateInput
            value={customDateRange.end}
            onChange={(newDate: any) =>
              handleCustomDateChange(customDateRange.start, newDate)
            }
          />
        </div>
      )}
    </div>
  )
}

function DateInput({ value, onChange }: any) {
  return (
    <input
      type="date"
      value={value}
      onChange={e => onChange(e.target.value)}
      className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
    />
  )
}

function ButtonGroup({ options, selectedValue, onChange }: any) {
  return (
    <div className="flex flex-wrap gap-2">
      {options.map((option: any) => (
        <Button
          key={option.value}
          variant={selectedValue === option.value ? "default" : "outline"}
          size="sm"
          className={`text-xs font-medium transition-all duration-200 ${
            selectedValue === option.value
              ? "bg-blue-500 text-white hover:bg-blue-600"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
          }`}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </Button>
      ))}
    </div>
  )
}

export const handleClearAllFilters = (
  setFilters: (filters: Filters) => void
) => {
  setFilters({
    sentiment: "all",
    model: "all",
    plugin: "all",
    rag: "all",
    reviewed: "all",
    date: "all"
  })
}
