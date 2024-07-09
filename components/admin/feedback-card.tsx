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
    <Card className="flex flex-col bg-white p-6 shadow-md transition-shadow duration-300 hover:shadow-lg dark:bg-gray-800">
      <div className="mb-4 flex items-start justify-between">
        <div className="flex items-center space-x-3">
          {feedback.feedback === "good" ? (
            <IconThumbUpFilled className="size-6 text-green-500" />
          ) : (
            <IconThumbDownFilled className="size-6 text-red-500" />
          )}
          <span className="text-lg font-semibold">{feedback.feedback}</span>
        </div>
        <span className="text-sm text-gray-500 dark:text-gray-400">
          {new Date(feedback.created_at).toLocaleString(undefined, {
            year: "numeric",
            month: "short",
            day: "numeric",
            hour: "2-digit",
            minute: "2-digit"
          })}
        </span>
      </div>
      <div className="mb-4 flex flex-wrap gap-2">
        <Badge variant="secondary" className="px-2 py-1 text-xs">
          {feedback.model}
        </Badge>
        {feedback.plugin && (
          <Badge variant="outline" className="px-2 py-1 text-xs">
            {feedback.plugin}
          </Badge>
        )}
        {feedback.rag_used && (
          <Badge variant="outline" className="px-2 py-1 text-xs">
            RAG
          </Badge>
        )}
      </div>
      {feedback.reason && (
        <p className="mb-4 text-gray-600 dark:text-gray-300">
          Reason: {feedback.reason}
        </p>
      )}
      {feedback.detailed_feedback && (
        <div className="mb-4">
          <h4 className="mb-2 text-base font-medium">Detailed Feedback:</h4>
          <p
            className="line-clamp-3 text-sm text-gray-600 dark:text-gray-300"
            title={feedback.detailed_feedback}
          >
            {feedback.detailed_feedback}
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
        >
          View Details
          <IconArrowRight className="ml-2 size-4" />
        </Button>
      </div>
    </Card>
  )
}
