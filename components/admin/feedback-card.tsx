import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import {
  IconArrowRight,
  IconThumbDownFilled,
  IconThumbUpFilled
} from "@tabler/icons-react"
import { FeedbackWithReview } from "@/types/feedback"

type FeedbackCardProps = {
  feedback: FeedbackWithReview
  onReviewToggle: (id: string) => void
  onViewDetails: (id: string) => void
}

export function FeedbackCard({
  feedback,
  onReviewToggle,
  onViewDetails
}: FeedbackCardProps) {
  return (
    <Card className="group flex flex-col bg-white p-6 shadow-sm transition-all duration-300 hover:shadow-md dark:bg-gray-800">
      <div className="mb-4 flex items-start justify-between">
        <div className="flex items-center space-x-3">
          {feedback.feedback === "good" ? (
            <IconThumbUpFilled className="size-8 rounded-full bg-green-100 p-1.5 text-green-600 dark:bg-green-900 dark:text-green-300" />
          ) : (
            <IconThumbDownFilled className="size-8 rounded-full bg-red-100 p-1.5 text-red-600 dark:bg-red-900 dark:text-red-300" />
          )}
          <span className="text-xl font-semibold capitalize">
            {feedback.feedback}
          </span>
        </div>
        <span className="text-sm font-medium text-gray-500 dark:text-gray-400">
          {formatDate(feedback.created_at)}
        </span>
      </div>
      <div className="mb-4 flex flex-wrap gap-2">
        <Badge
          variant="secondary"
          className="bg-gray-200 px-2 py-1 text-xs font-medium text-gray-800 dark:bg-gray-700 dark:text-gray-200"
        >
          {feedback.model}
        </Badge>
        {feedback.plugin && feedback.plugin !== "none" && (
          <Badge
            variant="outline"
            className="border-gray-300 px-2 py-1 text-xs font-medium text-gray-700 dark:border-gray-600 dark:text-gray-300"
          >
            {feedback.plugin}
          </Badge>
        )}
        {feedback.rag_used && (
          <Badge
            variant="outline"
            className="border-gray-300 px-2 py-1 text-xs font-medium text-gray-700 dark:border-gray-600 dark:text-gray-300"
          >
            RAG
          </Badge>
        )}
        {feedback.reason && (
          <Badge
            variant="outline"
            className="border-gray-300 px-2 py-1 text-xs font-medium text-gray-700 dark:border-gray-600 dark:text-gray-300"
          >
            {feedback.reason}
          </Badge>
        )}
      </div>
      {feedback.detailed_feedback && (
        <div className="mb-4 rounded-md p-3">
          <p
            className="line-clamp-3 italic text-gray-700 dark:text-gray-300"
            title={feedback.detailed_feedback}
          >
            &quot;{feedback.detailed_feedback}&quot;
          </p>
        </div>
      )}
      <div className="mt-auto flex items-center justify-between border-t border-gray-200 pt-4 dark:border-gray-700">
        <Label className="flex cursor-pointer items-center space-x-2">
          <Checkbox
            checked={feedback.feedback_reviews !== null}
            onCheckedChange={() => onReviewToggle(feedback.id)}
          />
          <span className="text-sm font-medium">Reviewed</span>
        </Label>
        <Button
          variant="ghost"
          size="sm"
          onClick={() => onViewDetails(feedback.id)}
          className="group-hover:bg-gray-100 dark:group-hover:bg-gray-700"
        >
          View Details
          <IconArrowRight className="ml-2 size-4 transition-transform group-hover:translate-x-1" />
        </Button>
      </div>
    </Card>
  )
}

function formatDate(date: string): string {
  return new Date(date).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  })
}
