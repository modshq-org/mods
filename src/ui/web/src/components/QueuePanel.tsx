import { useState, useCallback, useEffect } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  ChevronDownIcon,
  Loader2Icon,
  SparklesIcon,
  PencilIcon,
  Trash2Icon,
  XIcon,
} from 'lucide-react'
import { api, type QueueStatus } from '../api'
import { STALE_REALTIME } from '@/lib/query-keys'
import { useSSE } from '../hooks/useSSE'

export function QueuePanel() {
  const queryClient = useQueryClient()
  const [expanded, setExpanded] = useState(false)
  const [step, setStep] = useState<number>()
  const [totalSteps, setTotalSteps] = useState<number>()
  // Keep pill visible briefly after completion
  const [showDone, setShowDone] = useState(false)
  const [wasRunning, setWasRunning] = useState(false)

  const { data: qs } = useQuery<QueueStatus>({
    queryKey: ['generate-queue'],
    queryFn: api.queueStatus,
    refetchInterval: 2000,
    staleTime: STALE_REALTIME,
  })

  const isRunning = qs?.running ?? false
  const current = qs?.current
  const queue = qs?.queue ?? []

  // Track transitions from running → idle for "done" flash
  useEffect(() => {
    if (isRunning) {
      setWasRunning(true)
      setShowDone(false)
    } else if (wasRunning) {
      setWasRunning(false)
      setShowDone(true)
      const t = setTimeout(() => setShowDone(false), 3000)
      return () => clearTimeout(t)
    }
  }, [isRunning, wasRunning])

  // SSE for live step progress
  const handleSSEMessage = useCallback(
    (message: string) => {
      const lower = message.toLowerCase()

      // Step progress
      try {
        const parsed = JSON.parse(message)
        if (parsed.step != null && parsed.total_steps != null) {
          setStep(parsed.step)
          setTotalSteps(parsed.total_steps)
        }
      } catch {
        // not JSON
      }

      // Queue changes → refetch
      if (lower === 'queue:empty' || lower.startsWith('queue:')) {
        void queryClient.invalidateQueries({ queryKey: ['generate-queue'] })
      }

      // Completion → reset progress, refetch
      if (lower.includes('completed') || lower.includes('done')) {
        setStep(undefined)
        setTotalSteps(undefined)
        void queryClient.invalidateQueries({ queryKey: ['generate-queue'] })
      }
    },
    [queryClient],
  )

  useSSE(isRunning ? '/api/generate/stream' : null, handleSSEMessage)

  // Reset progress when idle
  useEffect(() => {
    if (!isRunning) {
      setStep(undefined)
      setTotalSteps(undefined)
    }
  }, [isRunning])

  // Close expanded panel when queue goes idle
  useEffect(() => {
    if (!isRunning && queue.length === 0) setExpanded(false)
  }, [isRunning, queue.length])

  // Nothing to show
  if (!isRunning && !showDone) return null

  const progressPct =
    step != null && totalSteps != null && totalSteps > 0
      ? Math.min(100, (step / totalSteps) * 100)
      : undefined

  const snippet = (text: string, max = 50) =>
    text.length > max ? text.slice(0, max) + '\u2026' : text

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col items-end gap-2">
      {/* Expanded panel */}
      {expanded && isRunning && (
        <div className="w-80 overflow-hidden rounded-lg border border-border/50 bg-[#0e0e18]/95 shadow-2xl backdrop-blur">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-border/30 px-3 py-2">
            <span className="text-xs font-medium text-foreground">
              Generation Queue
            </span>
            <button
              onClick={() => setExpanded(false)}
              className="rounded p-0.5 text-muted-foreground hover:text-foreground"
            >
              <ChevronDownIcon className="size-3.5" />
            </button>
          </div>

          {/* Current job */}
          {current && (
            <div className="border-b border-border/20 px-3 py-2.5">
              <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-muted-foreground/60">
                <Loader2Icon className="size-3 animate-spin text-primary" />
                {current.job_type === 'edit' ? 'Editing' : 'Generating'}
              </div>
              <p className="mt-1 text-xs leading-snug text-foreground/80 line-clamp-2">
                {snippet(current.prompt, 100)}
              </p>
              <p className="mt-0.5 flex items-center gap-1.5 text-[10px] text-muted-foreground">
                {current.job_type === 'edit' ? (
                  <PencilIcon className="size-2.5" />
                ) : (
                  <SparklesIcon className="size-2.5" />
                )}
                {current.model_id}
              </p>
              {progressPct != null && (
                <div className="mt-1.5 flex items-center gap-2">
                  <div className="h-1 flex-1 overflow-hidden rounded-full bg-secondary">
                    <div
                      className="h-full rounded-full bg-primary transition-all duration-300"
                      style={{ width: `${progressPct}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-muted-foreground/60">
                    {step}/{totalSteps}
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Queued items */}
          {queue.length > 0 && (
            <div className="max-h-48 overflow-y-auto">
              {queue.map((job, i) => (
                <div
                  key={i}
                  className="flex items-start gap-2 border-b border-border/10 px-3 py-2 last:border-0"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] font-medium text-muted-foreground/40">
                        #{i + 1}
                      </span>
                      <span className="text-[10px] text-muted-foreground/40">
                        {job.model_id}
                      </span>
                    </div>
                    <p className="text-xs text-foreground/60 line-clamp-1">
                      {snippet(job.prompt, 60)}
                    </p>
                  </div>
                  <button
                    onClick={async () => {
                      await api.cancelQueueItem(i)
                      void queryClient.invalidateQueries({
                        queryKey: ['generate-queue'],
                      })
                    }}
                    className="mt-0.5 shrink-0 rounded p-0.5 text-muted-foreground/30 transition-colors hover:text-destructive"
                    title="Remove from queue"
                  >
                    <XIcon className="size-3" />
                  </button>
                </div>
              ))}
            </div>
          )}

          {/* Footer: clear all */}
          {queue.length > 1 && (
            <div className="border-t border-border/30 px-3 py-1.5">
              <button
                onClick={async () => {
                  await api.clearQueue()
                  void queryClient.invalidateQueries({
                    queryKey: ['generate-queue'],
                  })
                }}
                className="flex items-center gap-1 text-[10px] text-muted-foreground/60 transition-colors hover:text-destructive"
              >
                <Trash2Icon className="size-3" />
                Clear all queued
              </button>
            </div>
          )}

          {/* Empty queue message */}
          {queue.length === 0 && current && (
            <div className="px-3 py-2 text-[10px] text-muted-foreground/40">
              Queue empty &mdash; add more from the Generate tab
            </div>
          )}
        </div>
      )}

      {/* Floating pill */}
      {showDone && !isRunning ? (
        <button
          onClick={() => setShowDone(false)}
          className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-[#0e0e18]/95 px-3 py-2 shadow-lg backdrop-blur transition-all"
        >
          <div className="size-2 rounded-full bg-emerald-400" />
          <span className="text-xs text-emerald-300">Done</span>
        </button>
      ) : isRunning ? (
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex items-center gap-2 rounded-full border border-border/50 bg-[#0e0e18]/95 px-3 py-2 shadow-lg backdrop-blur transition-all hover:border-border/70"
        >
          {/* Progress ring or spinner */}
          {progressPct != null ? (
            <svg className="size-4 -rotate-90" viewBox="0 0 20 20">
              <circle
                cx="10"
                cy="10"
                r="8"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="text-secondary"
              />
              <circle
                cx="10"
                cy="10"
                r="8"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeDasharray={`${(progressPct / 100) * 50.3} 50.3`}
                className="text-primary transition-all duration-300"
              />
            </svg>
          ) : (
            <Loader2Icon className="size-3.5 animate-spin text-primary" />
          )}
          <span className="text-xs text-foreground/80">
            {current?.job_type === 'edit' ? 'Editing' : 'Generating'}
            {progressPct != null && ` ${Math.round(progressPct)}%`}
          </span>
          {queue.length > 0 && (
            <span className="rounded-full bg-primary/20 px-1.5 py-0.5 text-[10px] font-medium text-primary">
              +{queue.length}
            </span>
          )}
        </button>
      ) : null}
    </div>
  )
}
