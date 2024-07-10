import { Button } from "@/components/ui/button"
import { useMemo } from "react"

interface PaginatorProps {
  currentPage: number
  totalPages: number
  onPageChange: (page: number) => void
}

export const Paginator: React.FC<PaginatorProps> = ({
  currentPage,
  totalPages,
  onPageChange
}) => {
  const isPrevDisabled = useMemo(() => currentPage === 1, [currentPage])
  const isNextDisabled = useMemo(
    () => currentPage === totalPages,
    [currentPage, totalPages]
  )

  return (
    <div className="bg-background border-border fixed inset-x-0 bottom-0 flex justify-center space-x-2 border-t p-4">
      <Button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={isPrevDisabled}
      >
        Previous
      </Button>
      <span className="flex items-center">
        Page {currentPage} of {totalPages}
      </span>
      <Button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={isNextDisabled}
      >
        Next
      </Button>
    </div>
  )
}
