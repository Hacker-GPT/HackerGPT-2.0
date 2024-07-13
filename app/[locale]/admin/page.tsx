/**
 * v0 by Vercel.
 * @see https://v0.dev/t/mEphc160VQ3
 * Documentation: https://v0.dev/docs#integrating-generated-code-into-your-nextjs-app
 */
"use client"

import { FeedbackList } from "@/components/admin/feedback-list"
import { Paginator } from "@/components/admin/feedback-pagination"
import { FilterBar, handleClearAllFilters } from "@/components/admin/filter-bar"
import { FeedbackDetails } from "@/components/admin/feedback-details"
import { ChatbotUIContext } from "@/context/context"
import {
  deleteFeedbackReview,
  upsertFeedbackReview
} from "@/db/message-feedback"
import { getMessagesByFeedbackId } from "@/db/messages"
import { supabase } from "@/lib/supabase/browser-client"
import { Tables } from "@/supabase/types"
import { FeedbackWithReview } from "@/types/feedback"
import { isAfter, parseISO, subHours } from "date-fns"
import { useRouter } from "next/navigation"
import { useCallback, useContext, useEffect, useMemo, useState } from "react"

const ITEMS_PER_PAGE = 20

type RagDataType = {
  id: string
  question: string
  expanded_questions: string
  final_text: string
  num_tokens: number
}

export default function Component() {
  const router = useRouter()
  const { profile } = useContext(ChatbotUIContext)

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

  const fetchFeedbacks = useCallback(async () => {
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
      .order("created_at", { ascending: false })

    if (error) {
      console.error("Error fetching feedbacks:", error)
    } else {
      setFeedbacks(data || [])
    }
    setLoading(false)
  }, [])

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

  const filterOptions = useMemo(
    () => ({
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
      ],
      date: [
        { value: "all", label: "All Time", count: feedbacks.length },
        { value: "6h", label: "Last 6 hours", count: 0 },
        { value: "12h", label: "Last 12 hours", count: 0 },
        { value: "24h", label: "Last 24 hours", count: 0 },
        { value: "7d", label: "Last 7 days", count: 0 },
        { value: "custom", label: "Custom", count: 0 }
      ]
    }),
    [feedbacks, models]
  )

  const filteredFeedbacks = useMemo(() => {
    return feedbacks.filter(feedback => {
      if (
        filters.sentiment !== "all" &&
        feedback.feedback !== filters.sentiment
      )
        return false
      if (filters.model !== "all" && feedback.model !== filters.model)
        return false
      if (filters.plugin !== "all") {
        if (filters.plugin === "used" && feedback.plugin === "none")
          return false
        if (filters.plugin === "not-used" && feedback.plugin !== "none")
          return false
      }
      if (
        filters.rag !== "all" &&
        feedback.rag_used !== (filters.rag === "used")
      )
        return false
      if (
        filters.reviewed !== "all" &&
        !!feedback.feedback_reviews !== (filters.reviewed === "true")
      )
        return false
      if (filters.date !== "all") {
        if (filters.date === "custom") {
          const start = parseISO(customDateRange.start)
          const end = parseISO(customDateRange.end)
          const feedbackDate = new Date(feedback.created_at)
          if (feedbackDate < start || feedbackDate > end) return false
        } else {
          const [value, unit] = filters.date.split(/(\d+)/).filter(Boolean)
          const hoursAgo = unit === "h" ? parseInt(value) : parseInt(value) * 24
          if (
            !isAfter(
              new Date(feedback.created_at),
              subHours(new Date(), hoursAgo)
            )
          )
            return false
        }
      }
      return true
    })
  }, [feedbacks, filters, customDateRange])

  const totalPages = useMemo(
    () => Math.ceil(filteredFeedbacks.length / ITEMS_PER_PAGE),
    [filteredFeedbacks]
  )

  const paginatedFeedbacks = useMemo(() => {
    const startIndex = (currentPage - 1) * ITEMS_PER_PAGE
    return filteredFeedbacks.slice(startIndex, startIndex + ITEMS_PER_PAGE)
  }, [filteredFeedbacks, currentPage])

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
      if (!profile || !profile.id) return

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
          const data = await upsertFeedbackReview(id, profile.user_id, "")
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
    [feedbacks, profile]
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
        handleClearAllFilters={() => {
          handleClearAllFilters(setFilters)
          setCurrentPage(1)
        }}
        customDateRange={customDateRange}
        handleCustomDateChange={handleCustomDateChange}
      />
      <FeedbackList
        feedbacks={paginatedFeedbacks}
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
