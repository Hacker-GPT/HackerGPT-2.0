import Modal from "@/components/chat/dialog-portal"
import { MessageMarkdown } from "@/components/messages/message-markdown"
import { Tables } from "@/supabase/types"
import { FeedbackWithReview } from "@/types/feedback"
import { IconLoader2, IconX } from "@tabler/icons-react"
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react"

type RagDataType = {
  id: string
  question: string
  expanded_questions: string
  final_text: string
  num_tokens: number
}

interface FeedbackDetailsProps {
  selectedFeedback: FeedbackWithReview | null
  chatMessages: Tables<"messages">[]
  ragData: RagDataType | null
  onClose: () => void
}

export const FeedbackDetails = memo(function FeedbackDetails({
    selectedFeedback,
    chatMessages,
    ragData,
    onClose
  }: FeedbackDetailsProps) {
    const modalRef = useRef<HTMLDivElement>(null)
  
    const handleClickOutside = useCallback((event: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(event.target as Node)) {
        onClose()
      }
    }, [onClose])
  
    useEffect(() => {
      document.addEventListener('mousedown', handleClickOutside)
      return () => document.removeEventListener('mousedown', handleClickOutside)
    }, [handleClickOutside])
  
    if (!selectedFeedback) return null
  
    return (
      <Modal isOpen={true}>
        <div className="fixed inset-0 z-50 bg-black/50 backdrop-blur-sm" aria-hidden="true" />
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <div 
            ref={modalRef} 
            className="bg-background w-full max-w-4xl overflow-hidden rounded-lg border shadow-lg"
            role="dialog"
            aria-labelledby="feedback-details-title"
          >
            <header className="flex items-center justify-between border-b bg-gray-50 p-4 dark:bg-gray-800">
              <h2 id="feedback-details-title" className="text-xl font-bold text-gray-800 dark:text-gray-200">Feedback Details</h2>
              <button
                onClick={onClose}
                className="rounded-full p-2 transition-colors hover:bg-gray-200 dark:hover:bg-gray-700"
                aria-label="Close"
              >
                <IconX size={20} />
              </button>
            </header>
            
            <div className="max-h-[calc(100vh-10rem)] overflow-y-auto">
              <div className="space-y-6 p-6">
                <ChatMessagesSection messages={chatMessages} />
                <RagDataContainer ragId={selectedFeedback.rag_id} ragData={ragData} />
                <ReviewDetailsSection review={selectedFeedback.feedback_reviews} />
              </div>
            </div>
          </div>
        </div>
      </Modal>
    )
  })
  
  const ChatMessagesSection = memo(function ChatMessagesSection({ messages }: { messages: Tables<"messages">[] }) {
    if (messages.length === 0) return null
    
    return (
      <Section title="Chat Messages">
        <ul className="space-y-4">
          {messages.map((message, index) => (
            <li key={index} className="rounded-lg bg-white p-4 shadow-sm dark:bg-gray-700">
              <p className="mb-2 font-semibold text-gray-700 dark:text-gray-300">{message.role}</p>
              <MessageMarkdown content={message.content} isAssistant={true} />
            </li>
          ))}
        </ul>
      </Section>
    )
  })
  
  const RagDataContainer = memo(function RagDataContainer({ ragId, ragData }: { ragId: string | null, ragData: RagDataType | null }) {
    if (!ragId) return null
    
    return ragData ? <RagDataSection ragData={ragData} /> : (
      <div className="flex h-32 items-center justify-center">
        <IconLoader2 className="animate-spin text-gray-500" size={32} />
      </div>
    )
  })
  
  const ReviewDetailsSection = memo(function ReviewDetailsSection({ review }: { review: FeedbackWithReview['feedback_reviews'] }) {
    if (!review) return null
    
    return (
      <Section title="Review Details">
        <div className="rounded-lg bg-white p-4 shadow-sm dark:bg-gray-700">
          <DataItem label="Reviewed By" value={review.reviewed_by} />
          <DataItem label="Reviewed At" value={new Date(review.reviewed_at).toLocaleString()} />
        </div>
      </Section>
    )
  })
      
      function Section({ title, children }: { title: string; children: React.ReactNode }) {
        return (
          <div className="mb-6">
            <h3 className="mb-3 text-lg font-bold text-gray-800 dark:text-gray-200">{title}</h3>
            {children}
          </div>
        )
      }
      
      function DataItem({ label, value }: { label: string; value: string }) {
        return (
          <div className="mb-2">
            <span className="font-semibold text-gray-700 dark:text-gray-300">{label}:</span>{' '}
            <span className="text-gray-600 dark:text-gray-400">{value}</span>
          </div>
        )
      }
      
      const RagDataSection = memo(function RagDataSection({ ragData }: { ragData: RagDataType }) {
        const [expandedQuestionsVisible, setExpandedQuestionsVisible] = useState(false)
        const finalTextParts = useMemo(() => ragData.final_text.split(/<FILE>|<\/FILE>/).filter(Boolean), [ragData.final_text])
      
        return (
          <Section title="RAG Data">
            <div className="space-y-4 rounded-lg bg-white p-4 shadow-sm dark:bg-gray-700">
              <DataItem label="Question" value={ragData.question} />
              <ExpandedQuestions
                questions={ragData.expanded_questions.split('\n')}
                visible={expandedQuestionsVisible}
                onToggle={() => setExpandedQuestionsVisible(prev => !prev)}
              />
              <FinalTextSection parts={finalTextParts} />
              <DataItem label="Num Tokens" value={ragData.num_tokens.toString()} />
            </div>
          </Section>
        )
      })
      
      const FinalTextSection = memo(function FinalTextSection({ parts }: { parts: string[] }) {
        return (
          <div>
            <h4 className="mb-2 font-semibold text-gray-700 dark:text-gray-300">Final Text</h4>
            {parts.map((part, index) => (
              <div key={index} className="mb-4">
                {index % 2 === 0 ? (
                  <p className="text-sm text-gray-600 dark:text-gray-400">{part.trim()}</p>
                ) : (
                  <div className="rounded-md bg-gray-100 p-3 dark:bg-gray-600">
                    <h5 className="mb-1 text-sm font-semibold text-gray-700 dark:text-gray-300">File Content:</h5>
                    <p className="text-sm text-gray-600 dark:text-gray-400">{part.trim()}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        )
      })
      
      const ExpandedQuestions = memo(function ExpandedQuestions({ 
        questions, 
        visible, 
        onToggle 
      }: { 
        questions: string[], 
        visible: boolean, 
        onToggle: () => void 
      }) {
        return (
          <div className="rounded-lg bg-gray-50 p-2 dark:bg-gray-600">
            <button
              className="flex w-full items-center justify-between font-semibold text-gray-700 transition-colors hover:text-gray-900 dark:text-gray-300 dark:hover:text-gray-100"
              onClick={onToggle}
              aria-expanded={visible}
            >
              <span>Expanded Questions</span>
              <span className="rounded-full bg-gray-200 px-2 py-1 text-sm dark:bg-gray-500">
                {visible ? 'Hide' : 'Show'}
              </span>
            </button>
            {visible && (
              <ul className="mt-3 space-y-2 text-sm text-gray-600 dark:text-gray-300">
                {questions.map((question, index) => (
                  <li key={index} className="relative pl-2">{question.trim()}</li>
                ))}
              </ul>
            )}
          </div>
        )
      })