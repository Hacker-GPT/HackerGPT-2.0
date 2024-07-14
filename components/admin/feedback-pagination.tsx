import { Button } from "@/components/ui/button"
import { useMemo } from "react"

interface PaginatorProps {
  currentPage: number
  onPageChange: (page: number) => void
  hasMore: boolean
}

export const Paginator: React.FC<PaginatorProps> = ({
  currentPage,
  onPageChange,
  hasMore
}) => {
  const isPrevDisabled = useMemo(() => currentPage === 1, [currentPage])

  return (
    <div className="bg-background border-border fixed inset-x-0 bottom-0 flex justify-center space-x-2 border-t p-4">
      <Button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={isPrevDisabled}
      >
        Previous
      </Button>
      <span className="flex items-center">Page {currentPage}</span>
      <Button onClick={() => onPageChange(currentPage + 1)} disabled={!hasMore}>
        Next
      </Button>
    </div>
  )
}
