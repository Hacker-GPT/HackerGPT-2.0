import { Button } from "@/components/ui/button"
import { FilterOptions, Filters } from "@/types/filters"

type FilterBarProps = {
  filterOptions: FilterOptions
  filters: Filters
  handleFilterChange: (key: string, value: any) => void
}

export function FilterBar({
  filterOptions,
  filters,
  handleFilterChange
}: FilterBarProps) {
  return (
    <div className="mb-6 rounded-lg bg-gray-100 p-4 dark:bg-gray-800">
      <h2 className="mb-3 text-lg font-semibold">Filters</h2>
      <div className="flex flex-wrap gap-4">
        {Object.entries(filterOptions).map(([key, options]) => (
          <div key={key} className="flex flex-col">
            <span className="mb-2 text-sm font-medium">
              {key.charAt(0).toUpperCase() + key.slice(1)}
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
                  className="text-xs"
                  onClick={() => handleFilterChange(key, option.value)}
                >
                  {option.label}
                  <span className="ml-1 text-xs text-gray-500">
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
