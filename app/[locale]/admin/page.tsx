/**
 * v0 by Vercel.
 * @see https://v0.dev/t/mEphc160VQ3
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */
"use client"

import { FeedbackList } from "@/components/admin/feedback-list"
import { FilterBar, handleClearAllFilters } from "@/components/admin/filter-bar"
import { ChatbotUIContext } from "@/context/context"
import { supabase } from "@/lib/supabase/browser-client"
import { useRouter } from "next/navigation"
import { useCallback, useContext, useEffect, useMemo, useState } from "react"
import { FeedbackWithReview } from "@/types/feedback"
import { Paginator } from "@/components/admin/feedback-pagination"

const ITEMS_PER_PAGE = 20

export default function Component() {
  const router = useRouter()

  const [filters, setFilters] = useState({
    sentiment: "all",
    model: "all",
    plugin: "all",
    rag: "all",
    reviewed: "all"
  })

  const [feedbacks, setFeedbacks] = useState<FeedbackWithReview[]>([])
  const [loading, setLoading] = useState(true)
  const [currentPage, setCurrentPage] = useState(1)
  const [totalItems, setTotalItems] = useState(0)

  const { profile } = useContext(ChatbotUIContext)

  useEffect(() => {
    ;(async () => {
      const session = (await supabase.auth.getSession()).data.session

      if (!session) {
        return router.push("/login")
      } else {
        await fetchFeedbacks(currentPage)
      }
    })()
  }, [currentPage])

  const fetchFeedbacks = useCallback(async (page: number) => {
    setLoading(true)

    const from = (page - 1) * ITEMS_PER_PAGE
    const to = from + ITEMS_PER_PAGE - 1

    const { data, error, count } = await supabase
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
      `,
        { count: "exact" }
      )
      .range(from, to)
      .order("created_at", { ascending: false })

    if (error) {
      console.error("Error fetching feedbacks:", error)
    } else {
      setFeedbacks(data || [])
      setTotalItems(count || 0)
    }
    setLoading(false)
  }, [])

  useEffect(() => {
    fetchFeedbacks(currentPage)
  }, [currentPage, fetchFeedbacks])

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

  const filterOptions = useMemo(() => ({
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
  }), [feedbacks, models])

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

  const handleFilterChange = useCallback((key: string, value: any) => {
    setFilters(prevFilters => ({
      ...prevFilters,
      [key]: prevFilters[key as keyof typeof prevFilters] === value ? "all" : value
    }))
  }, [])

  const handleReviewToggle = useCallback(async (id: string) => {
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
  }, [feedbacks, profile])

  const handleViewDetails = async (id: string) => {
    // Implement the logic to fetch and display details
    console.log(`Viewing details for feedback id: ${id}`)
  }

  const totalPages = useMemo(
    () => Math.ceil(totalItems / ITEMS_PER_PAGE),
    [totalItems]
  )

  const handlePageChange = useCallback((newPage: number) => {
    setCurrentPage(newPage)
  }, [])

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
    <div className="container mx-auto py-10 pb-20">
      <h1 className="mb-6 text-3xl font-bold">Feedback</h1>
      <FilterBar
        filterOptions={filterOptions}
        filters={filters}
        handleFilterChange={handleFilterChange}
        handleClearAllFilters={() => handleClearAllFilters(setFilters)}
      />
      <FeedbackList
        feedbacks={filteredFeedbacks}
        onReviewToggle={handleReviewToggle}
        onViewDetails={handleViewDetails}
      />
      <Paginator
        currentPage={currentPage}
        totalPages={totalPages}
        onPageChange={handlePageChange}
      />
    </div>
  )
}
