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

  const handleDateButtonClick = (type: "start" | "end") => {
    const input = document.createElement("input")
    input.type = "date"
    input.value = customDateRange[type]
    input.onchange = e => {
      const newDate = (e.target as HTMLInputElement).value
      if (type === "start") {
        handleCustomDateChange(newDate, customDateRange.end)
      } else {
        handleCustomDateChange(customDateRange.start, newDate)
      }
    }
    input.click()
  }

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
              <div className="flex gap-2">
                <div className="relative">
                  <select
                    value={filters.date}
                    onChange={e => {
                      handleFilterChange(key, e.target.value)
                      setShowCustomDate(e.target.value === "custom")
                    }}
                    className="appearance-none rounded-md border border-gray-300 bg-white px-3 py-1.5 pr-8 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
                  >
                    {options.map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                  <ChevronDown className="pointer-events-none absolute right-2 top-1/2 size-4 -translate-y-1/2 text-gray-400" />
                </div>
                {showCustomDate && (
                  <div className="flex gap-2">
                    <input
                      type="date"
                      value={customDateRange.start}
                      onChange={e =>
                        handleCustomDateChange(
                          e.target.value,
                          customDateRange.end
                        )
                      }
                      className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
                    />
                    <input
                      type="date"
                      value={customDateRange.end}
                      onChange={e =>
                        handleCustomDateChange(
                          customDateRange.start,
                          e.target.value
                        )
                      }
                      className="rounded-md border border-gray-300 bg-white px-3 py-1.5 text-sm font-medium text-gray-700 shadow-sm hover:bg-gray-50 focus:border-blue-500 focus:outline-none focus:ring-1 focus:ring-blue-500 dark:border-gray-600 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {options.map(option => (
                  <Button
                    key={option.value}
                    variant={
                      filters[key as keyof Filters] === option.value
                        ? "default"
                        : "outline"
                    }
                    size="sm"
                    className={`text-xs font-medium transition-all duration-200 ${
                      filters[key as keyof Filters] === option.value
                        ? "bg-blue-500 text-white hover:bg-blue-600"
                        : "bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600"
                    }`}
                    onClick={() => handleFilterChange(key, option.value)}
                  >
                    {option.label}
                    <span
                      className={`ml-1 text-xs ${
                        filters[key as keyof Filters] === option.value
                          ? "text-blue-200"
                          : "text-gray-500 dark:text-gray-400"
                      }`}
                    >
                      ({option.count})
                    </span>
                  </Button>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
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
