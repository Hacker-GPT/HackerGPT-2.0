import { FeedbackWithReview } from "@/types/feedback"
import { FeedbackCard } from "@/components/admin/feedback-card"

type FeedbackListProps = {
  feedbacks: FeedbackWithReview[]
  onReviewToggle: (id: string) => void
  onViewDetails: (id: string) => void
}

export function FeedbackList({
  feedbacks,
  onReviewToggle,
  onViewDetails
}: FeedbackListProps) {
  return (
    <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 xl:grid-cols-3">
      {feedbacks.map(feedback => (
        <FeedbackCard
          feedback={feedback}
          onReviewToggle={onReviewToggle}
          onViewDetails={onViewDetails}
        />
      ))}
    </div>
  )
}
