import { Button } from "@/components/ui/button"
import { FilterOptions, Filters } from "@/types/filters"

type FilterBarProps = {
  filterOptions: FilterOptions
  filters: Filters
  handleFilterChange: (key: string, value: any) => void
  handleClearAllFilters: () => void
}

export function FilterBar({
  filterOptions,
  filters,
  handleFilterChange,
  handleClearAllFilters
}: FilterBarProps) {
  return (
    <div className="mb-8 rounded-xl bg-white p-6 shadow-md transition-shadow duration-300 hover:shadow-lg dark:bg-gray-800">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-800 dark:text-white">Filters</h2>
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
              {key}
            </span>
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
                      ? 'bg-blue-500 text-white hover:bg-blue-600'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600'
                  }`}
                  onClick={() => handleFilterChange(key, option.value)}
                >
                  {option.label}
                  <span className={`ml-1 text-xs ${
                    filters[key as keyof Filters] === option.value
                      ? 'text-blue-200'
                      : 'text-gray-500 dark:text-gray-400'
                  }`}>
                    ({option.count})
                  </span>
                </Button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}