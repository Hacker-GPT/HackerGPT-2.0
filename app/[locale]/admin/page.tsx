/**
 * v0 by Vercel.
 * @see https://v0.dev/t/mEphc160VQ3
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */
"use client"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Checkbox } from "@/components/ui/checkbox"
import { Label } from "@/components/ui/label"
import { ChatbotUIContext } from "@/context/context"
import { supabase } from "@/lib/supabase/browser-client"
import { Database } from "@/supabase/types"
import {
  IconArrowRight,
  IconThumbDownFilled,
  IconThumbUpFilled
} from "@tabler/icons-react"
import { useRouter } from "next/navigation"
import { useContext, useEffect, useMemo, useState } from "react"

type Feedback = Database["public"]["Tables"]["feedback"]["Row"]

export default function Component() {
  const router = useRouter()

  const [filters, setFilters] = useState({
    sentiment: "all",
    model: "all",
    plugin: "all",
    rag: "all",
    reviewed: "all"
  })

  type FeedbackWithReview = Feedback & {
    feedback_reviews: {
      id: string
      reviewed_by: string
      reviewed_at: string
      notes: string | null
    } | null
  }

  const [feedbacks, setFeedbacks] = useState<FeedbackWithReview[]>([])
  const [loading, setLoading] = useState(true)

  const { profile } = useContext(ChatbotUIContext)

  useEffect(() => {
    ;(async () => {
      const session = (await supabase.auth.getSession()).data.session

      if (!session) {
        return router.push("/login")
      } else {
        await fetchFeedbacks()
      }
    })()
  }, [])

  const fetchFeedbacks = async () => {
    setLoading(true)

    const { data, error } = await supabase
      .from("feedback")
      .select(
        `
        *,
        feedback_reviews (
          id,
          reviewed_by,
          reviewed_at,
          notes
        )
      `
      )
      .limit(100)
      .order("created_at", { ascending: false })

    console.log(data)

    if (error) {
      console.error("Error fetching feedbacks:", error)
    } else {
      setFeedbacks(data || [])
    }
    setLoading(false)
  }

  const models = useMemo(() => {
    console.log(feedbacks)
    const uniqueModels = [...new Set(feedbacks.map(f => f.model))]
    return uniqueModels
      .filter(model => model !== null)
      .map(model => ({
        value: model,
        label: model,
        count: feedbacks.filter(f => f.model === model).length
      }))
  }, [feedbacks]) as { value: string; label: string; count: number }[]

  const filterOptions = {
    sentiment: [
      {
        value: "good",
        label: "Good",
        count: feedbacks.filter(f => f.feedback === "good").length
      },
      {
        value: "bad",
        label: "Bad",
        count: feedbacks.filter(f => f.feedback === "bad").length
      }
    ],
    model: models,
    plugin: [
      {
        value: "used",
        label: "Plugin Used",
        count: feedbacks.filter(f => f.plugin !== "none").length
      },
      {
        value: "not-used",
        label: "No Plugin",
        count: feedbacks.filter(f => f.plugin === "none").length
      }
    ],
    rag: [
      {
        value: "used",
        label: "RAG Used",
        count: feedbacks.filter(f => f.rag_used).length
      },
      {
        value: "not-used",
        label: "No RAG",
        count: feedbacks.filter(f => !f.rag_used).length
      }
    ],
    reviewed: [
      {
        value: "true",
        label: "Reviewed",
        count: feedbacks.filter(f => f.feedback_reviews).length
      },
      {
        value: "false",
        label: "Not Reviewed",
        count: feedbacks.filter(f => !f.feedback_reviews).length
      }
    ]
  }

  const filteredFeedbacks = useMemo(() => {
    return feedbacks.filter(feedback => {
      if (
        filters.sentiment !== "all" &&
        feedback.feedback !== filters.sentiment
      ) {
        return false
      }
      if (filters.model !== "all" && feedback.model !== filters.model) {
        return false
      }
      if (filters.plugin !== "all") {
        if (filters.plugin === "used" && feedback.plugin === "none")
          return false
        if (filters.plugin === "not-used" && feedback.plugin !== "none")
          return false
      }
      if (
        filters.rag !== "all" &&
        feedback.rag_used !== (filters.rag === "used")
      ) {
        return false
      }
      if (filters.reviewed !== "all" && feedback.feedback_reviews) {
        return false
      }
      return true
    })
  }, [feedbacks, filters])

  const handleFilterChange = (key: string, value: any) => {
    setFilters(prevFilters => ({
      ...prevFilters,
      [key]:
        prevFilters[key as keyof typeof prevFilters] === value ? "all" : value
    }))
  }

  const handleReviewToggle = async (id: string) => {
    const feedbackToUpdate = feedbacks.find(f => f.id === id)
    if (!feedbackToUpdate) return
    if (!profile || !profile.id) return

    if (feedbackToUpdate.feedback_reviews) {
      const { data, error } = await supabase
        .from("feedback_reviews")
        .delete()
        .eq("feedback_id", id)

      if (error) {
        console.error("Error updating feedback review:", error)
      } else {
        setFeedbacks(prevFeedbacks =>
          prevFeedbacks.map(f =>
            f.id === id ? { ...f, feedback_reviews: null } : f
          )
        )

        return
      }
    }

    const { data, error } = await supabase
      .from("feedback_reviews")
      .upsert({
        feedback_id: id,
        reviewed_by: profile.user_id,
        reviewed_at: new Date().toISOString(),
        notes: ""
      })
      .select()

    if (error) {
      console.error("Error updating feedback review:", error)
    } else if (data) {
      setFeedbacks(prevFeedbacks =>
        prevFeedbacks.map(f =>
          f.id === id ? { ...f, feedback_reviews: data[0] } : f
        )
      )
    }
  }

  const handleViewDetails = async (id: string) => {
    // Implement the logic to fetch and display details
    console.log(`Viewing details for feedback id: ${id}`)
  }

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        Loading...
      </div>
    )
  }

  if (profile && profile?.role !== "moderator") {
    router.push("/")
    return null
  }

  return (
    <div className="container mx-auto py-10">
      <div className="mb-6">
        <h1 className="text-2xl font-bold">Feedback Reviews</h1>
        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {Object.entries(filterOptions).map(([key, options]) => (
              <div key={key} className="flex items-center space-x-2">
                {options.map(option => (
                  <Button
                    key={option.value}
                    variant={
                      filters[key as keyof typeof filters] === option.value
                        ? "default"
                        : "outline"
                    }
                    className="rounded-md px-2 py-1 text-xs"
                    onClick={() => handleFilterChange(key, option.value)}
                  >
                    {option.label}{" "}
                    <span className="text-muted-foreground ml-1">
                      ({option.count})
                    </span>
                  </Button>
                ))}
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 gap-6 sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3">
        {filteredFeedbacks.map(feedback => (
          <Card
            key={feedback.id}
            className="flex flex-col bg-white p-4 shadow-md dark:bg-gray-800"
          >
            <div className="mb-4 flex items-start justify-between">
              <div className="flex items-center space-x-2">
                {feedback.feedback === "good" ? (
                  <IconThumbUpFilled className="size-6 text-green-500" />
                ) : (
                  <IconThumbDownFilled className="size-6 text-red-500" />
                )}
                <span className="font-medium">{feedback.feedback}</span>
                <span className="text-muted-foreground text-sm">
                  {new Date(feedback.created_at).toLocaleString(undefined, {
                    year: "numeric",
                    month: "numeric",
                    day: "numeric",
                    hour: "2-digit",
                    minute: "2-digit"
                  })}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-muted-foreground">{feedback.model}</span>
                {feedback.plugin && (
                  <Badge variant="secondary" className="px-2 py-1 text-xs">
                    {feedback.plugin}
                  </Badge>
                )}
                {feedback.rag_used && (
                  <Badge variant="outline" className="px-2 py-1 text-xs">
                    RAG
                  </Badge>
                )}
              </div>
            </div>
            {feedback.reason && (
              <p className="text-muted-foreground mb-4">
                Reason: {feedback.reason}
              </p>
            )}

            {feedback.detailed_feedback && (
              <div className="mb-4">
                <h4
                  className="mb-2 truncate text-lg font-semibold"
                  title={feedback.detailed_feedback}
                >
                  {feedback.detailed_feedback.length > 50
                    ? feedback.detailed_feedback.substring(0, 50) + "..."
                    : feedback.detailed_feedback}
                </h4>
              </div>
            )}

            <div className="mt-auto flex items-center justify-between">
              <Label className="flex items-center space-x-2">
                <Checkbox
                  checked={feedback.feedback_reviews !== null}
                  onCheckedChange={() => handleReviewToggle(feedback.id)}
                />
                <span>Reviewed</span>
              </Label>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => handleViewDetails(feedback.id)}
              >
                <IconArrowRight className="size-4" />
              </Button>
            </div>
          </Card>
        ))}
      </div>
    </div>
  )
}
