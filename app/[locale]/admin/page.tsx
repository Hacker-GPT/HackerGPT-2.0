/**
 * v0 by Vercel.
 * @see https://v0.dev/t/mEphc160VQ3
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */
"use client"

import { FeedbackDetails } from "@/components/admin/feedback-details"
import { FeedbackList } from "@/components/admin/feedback-list"
import { Paginator } from "@/components/admin/feedback-pagination"
import { FilterBar, handleClearAllFilters } from "@/components/admin/filter-bar"
import { ChatbotUIContext } from "@/context/context"
import {
  deleteFeedbackReview,
  getFeedbackSummary,
  upsertFeedbackReview
} from "@/db/message-feedback"
import { getMessagesByFeedbackId } from "@/db/messages"
import { supabase } from "@/lib/supabase/browser-client"
import { Tables } from "@/supabase/types"
import { FeedbackWithReview } from "@/types/feedback"
import { FilterOption, FilterOptions } from "@/types/filters"
import { isAfter, parseISO, set, subHours } from "date-fns"
import { useRouter } from "next/navigation"
import { useCallback, useContext, useEffect, useMemo, useState } from "react"

// For testing purposes, display 2 items per page. Reset to 20 when ready.
const ITEMS_PER_PAGE = 2

type RagDataType = {
  id: string
  question: string
  expanded_questions: string
  final_text: string
  num_tokens: number
}

export default function Component() {
  const router = useRouter()
  const { userRole } = useContext(ChatbotUIContext)

  const [filters, setFilters] = useState({
    sentiment: "all",
    model: "all",
    plugin: "all",
    rag: "all",
    reviewed: "all",
    date: "all"
  })
  const [customDateRange, setCustomDateRange] = useState({ start: "", end: "" })
  const [feedbacks, setFeedbacks] = useState<FeedbackWithReview[]>([])
  const [loading, setLoading] = useState(true)
  const [currentPage, setCurrentPage] = useState(1)
  const [selectedFeedback, setSelectedFeedback] =
    useState<FeedbackWithReview | null>(null)
  const [chatMessages, setChatMessages] = useState<Tables<"messages">[]>([])
  const [ragData, setRagData] = useState<RagDataType | null>(null)
  const [totalPages, setTotalPages] = useState(0)
  const [filterOptions, setFilterOptions] = useState<FilterOptions>({
    sentiment: [],
    model: [],
    plugin: [],
    rag: [],
    reviewed: [],
    date: []
  })

  useEffect(() => {
    ;(async () => {
      const session = (await supabase.auth.getSession()).data.session
      if (!session) {
        return router.push("/login")
      }
    })()
  }, [])

  useEffect(() => {
    setChatMessages([])
    setRagData(null)
    async function fetchMessages() {
      if (selectedFeedback && selectedFeedback.chat_id) {
        const fetchedMessages = await getMessagesByFeedbackId(
          selectedFeedback.id
        )

        if (!fetchedMessages) return

        const reformatedMessages = fetchedMessages.slice(-4)
        setChatMessages(reformatedMessages)
      }
    }
    async function fetchRagData() {
      if (!selectedFeedback || !selectedFeedback.rag_id) return
      const response = await fetch(
        `/api/admin/ragData?id=${selectedFeedback?.rag_id}`
      )
      const ragData = await response.json()
      setRagData(ragData)
    }
    fetchMessages()
    fetchRagData()
  }, [selectedFeedback])

  useEffect(() => {
    fetchFeedbacks()
  }, [currentPage, filters, customDateRange])

  const fetchFeedbacks = useCallback(async () => {
    setLoading(true)
    const { sentiment, model, plugin, rag, reviewed, date } = filters
    let query = supabase
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
      .order("created_at", { ascending: false })

    if (sentiment !== "all") {
      query = query.eq("feedback", sentiment)
    }
    if (model !== "all") {
      query = query.eq("model", model)
    }
    if (plugin !== "all") {
      if (plugin === "not-used") {
        query = query.eq("plugin", "none")
      } else {
        query = query.not("plugin", "eq", "none")
      }
    }
    if (rag !== "all") {
      query = query.eq("rag_used", rag === "used")
    }
    if (reviewed !== "all") {
      query = query.eq("reviewed", reviewed === "true")
    }
    if (date !== "all") {
      const now = new Date()
      let startDate

      switch (date) {
        case "6h":
          startDate = new Date(now.getTime() - 6 * 60 * 60 * 1000)
          break
        case "12h":
          startDate = new Date(now.getTime() - 12 * 60 * 60 * 1000)
          break
        case "24h":
          startDate = new Date(now.getTime() - 24 * 60 * 60 * 1000)
          break
        case "7d":
          startDate = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000)
          break
        default:
          startDate = new Date(date)
      }

      query = query.gte("created_at", startDate.toISOString())
    }

    const from = (currentPage - 1) * ITEMS_PER_PAGE
    const to = from + ITEMS_PER_PAGE - 1
    query = query.range(from, to)

    const { data, error, count } = await query

    if (error) {
      console.error("Error fetching feedbacks:", error)
    } else {
      setFeedbacks(data || [])
    }
    setLoading(false)
  }, [currentPage, filters])

  const models = useMemo(() => {
    const uniqueModels = [...new Set(feedbacks.map(f => f.model))]
    return uniqueModels
      .filter((model): model is string => model !== null)
      .map(model => ({
        value: model,
        label: model,
        count: feedbacks.filter(f => f.model === model).length
      }))
  }, [feedbacks])

  useEffect(() => {
    async function fetchFilterOptions() {
      const counts = await getFeedbackSummary()
      setFilterOptions({
        sentiment: [
          {
            value: "good",
            label: "Good",
            count: counts.feedback_good
          },
          {
            value: "bad",
            label: "Bad",
            count: counts.feedback_bad
          }
        ] as FilterOption[],
        model: models as FilterOption[],
        plugin: [
          {
            value: "used",
            label: "Plugin Used",
            count: counts.plugin_used
          },
          {
            value: "not-used",
            label: "No Plugin",
            count: counts.plugin_not_used
          }
        ] as FilterOption[],
        rag: [
          {
            value: "used",
            label: "RAG Used",
            count: counts.rag_used
          },
          {
            value: "not-used",
            label: "No RAG",
            count: counts.rag_not_used
          }
        ] as FilterOption[],
        reviewed: [
          {
            value: "true",
            label: "Reviewed",
            count: counts.reviewed
          },
          {
            value: "false",
            label: "Not Reviewed",
            count: counts.not_reviewed
          }
        ] as FilterOption[],
        date: [
          { value: "all", label: "All Time", count: counts.all_time },
          { value: "6h", label: "Last 6 hours", count: counts.last_6h },
          { value: "12h", label: "Last 12 hours", count: counts.last_12h },
          { value: "24h", label: "Last 24 hours", count: counts.last_24h },
          { value: "7d", label: "Last 7 days", count: counts.last_7d },
          { value: "custom", label: "Custom", count: null }
        ] as FilterOption[]
      })
      setTotalPages(Math.ceil(counts.all_time / ITEMS_PER_PAGE))
    }
    fetchFilterOptions()
  }, [feedbacks, models])

  const handlePageChange = useCallback((newPage: number) => {
    setCurrentPage(newPage)
  }, [])

  const handleFilterChange = useCallback((key: string, value: any) => {
    setFilters(prevFilters => ({
      ...prevFilters,
      [key]:
        prevFilters[key as keyof typeof prevFilters] === value ? "all" : value
    }))
    setCurrentPage(1)
  }, [])

  const handleReviewToggle = useCallback(
    async (id: string) => {
      const feedbackToUpdate = feedbacks.find(f => f.id === id)
      if (!feedbackToUpdate) return
      if (!userRole || !userRole.role) return

      if (feedbackToUpdate.feedback_reviews) {
        try {
          await deleteFeedbackReview(feedbackToUpdate.feedback_reviews.id)
          setFeedbacks(prevFeedbacks =>
            prevFeedbacks.map(f =>
              f.id === id ? { ...f, feedback_reviews: null } : f
            )
          )
        } catch (error) {
          console.error("Error deleting feedback review:", error)
        }
      } else {
        try {
          const data = await upsertFeedbackReview(id, userRole.user_id, "")
          setFeedbacks(prevFeedbacks =>
            prevFeedbacks.map(f =>
              f.id === id ? { ...f, feedback_reviews: data[0] } : f
            )
          )
        } catch (error) {
          console.error("Error updating feedback review:", error)
        }
      }
    },
    [feedbacks, userRole]
  )

  const handleViewDetails = async (id: string) => {
    const feedback = feedbacks.find(f => f.id === id)
    if (feedback) {
      setSelectedFeedback(feedback)
    }
  }

  const handleCustomDateChange = (start: string, end: string) => {
    setCustomDateRange({ start, end })
    setFilters(prev => ({ ...prev, date: "custom" }))
    setCurrentPage(1)
  }

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        Loading...
      </div>
    )
  }

  if (!userRole || userRole?.role !== "moderator") {
    return (
      <div className="flex h-screen items-center justify-center">
        Loading...
      </div>
    )
  }

  return (
    <div className="container mx-auto py-10 pb-20">
      <h1 className="mb-6 text-3xl font-bold">Feedback</h1>
      <FilterBar
        filterOptions={filterOptions}
        filters={filters}
        handleFilterChange={handleFilterChange}
        handleClearAllFilters={() => {
          handleClearAllFilters(setFilters)
          setCurrentPage(1)
        }}
        customDateRange={customDateRange}
        handleCustomDateChange={handleCustomDateChange}
      />
      <FeedbackList
        feedbacks={feedbacks}
        onReviewToggle={handleReviewToggle}
        onViewDetails={handleViewDetails}
      />
      <Paginator
        currentPage={currentPage}
        totalPages={totalPages}
        onPageChange={handlePageChange}
      />
      <FeedbackDetails
        selectedFeedback={selectedFeedback}
        chatMessages={chatMessages}
        ragData={ragData}
        onClose={() => setSelectedFeedback(null)}
      />
    </div>
  )
}
