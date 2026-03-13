import { useMemo, useRef, useEffect, useState } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import {
  Plus,
  Clock,
  SlidersHorizontal,
  X,
  Zap,
  Terminal,
} from 'lucide-react'
import {
  api,
  type TrainingQueueItem,
} from '../api'
import { timeAgo, displayModelName } from '@/lib/utils'
import { STALE_FAST } from '@/lib/query-keys'
import { TrainingRunCard, SampleLightbox, NewTrainingForm } from './training'
import type { SampleLightboxImage } from './training'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type StatusFilter = 'all' | 'running' | 'completed' | 'interrupted' | 'failed'
type SortMode = 'recent' | 'name'

const FILTER_TABS: { key: StatusFilter; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'running', label: 'Running' },
  { key: 'completed', label: 'Completed' },
  { key: 'interrupted', label: 'Interrupted' },
  { key: 'failed', label: 'Failed' },
]

// Sentinel value for "new training form" selection
const NEW_RUN = '__new__'

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const STATUS_INFO: Record<string, { label: string; dot: string; badge: string }> = {
  running:     { label: 'Running',     dot: 'bg-emerald-400', badge: 'border-emerald-500/50 text-emerald-300' },
  completed:   { label: 'Completed',   dot: 'bg-primary',     badge: 'border-primary/50 text-primary' },
  interrupted: { label: 'Interrupted', dot: 'bg-amber-400',   badge: 'border-amber-500/50 text-amber-300' },
  failed:      { label: 'Failed',      dot: 'bg-red-400',     badge: 'border-red-500/50 text-red-300' },
  error:       { label: 'Failed',      dot: 'bg-red-400',     badge: 'border-red-500/50 text-red-300' },
  cancelled:   { label: 'Failed',      dot: 'bg-red-400',     badge: 'border-red-500/50 text-red-300' },
  unknown:     { label: 'Unknown',     dot: 'bg-muted-foreground/40', badge: 'border-border text-muted-foreground' },
}

function getStatusInfo(status: string) {
  return STATUS_INFO[status] ?? STATUS_INFO.unknown
}

/** Normalize status for filtering: error/cancelled -> failed */
function normalizeStatus(status: string): string {
  if (status === 'error' || status === 'cancelled') return 'failed'
  return status
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export function TrainingRuns() {
  const queryClient = useQueryClient()

  // UI state — '__new__' means show the new training form
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [sortBy, setSortBy] = useState<SortMode>('recent')
  const [lightboxImage, setLightboxImage] = useState<SampleLightboxImage | null>(null)

  const autoSelectedRef = useRef(false)

  // -----------------------------------------------------------------------
  // Queries
  // -----------------------------------------------------------------------

  const {
    data: runs = [],
    error: runsError,
    isLoading: runsLoading,
  } = useQuery({
    queryKey: ['runs'],
    queryFn: api.runs,
    staleTime: STALE_FAST,
  })

  const { data: statusRuns = [] } = useQuery({
    queryKey: ['status'],
    queryFn: api.status,
    refetchInterval: 5000,
  })

  const { data: trainingQueue = [] } = useQuery({
    queryKey: ['training-queue'],
    queryFn: api.trainingQueue,
    refetchInterval: 10_000,
  })

  // -----------------------------------------------------------------------
  // Derived state
  // -----------------------------------------------------------------------

  const runningNames = useMemo(
    () => new Set(statusRuns.filter((r) => r.is_running).map((r) => r.name)),
    [statusRuns],
  )

  /** Effective status per run: live process overrides DB status. */
  const effectiveStatus = useMemo(() => {
    const map = new Map<string, string>()
    for (const run of runs) {
      if (runningNames.has(run.name)) {
        map.set(run.name, 'running')
      } else if (run.status === 'running') {
        map.set(run.name, 'interrupted')
      } else if (run.status === 'unknown') {
        if (run.has_lora) {
          map.set(run.name, 'completed')
        } else {
          map.set(run.name, 'interrupted')
        }
      } else {
        map.set(run.name, run.status)
      }
    }
    return map
  }, [runs, runningNames])

  /** Status counts for filter tabs */
  const statusCounts = useMemo(() => {
    const counts: Record<string, number> = { all: runs.length, running: 0, completed: 0, interrupted: 0, failed: 0 }
    for (const run of runs) {
      const status = normalizeStatus(effectiveStatus.get(run.name) ?? run.status)
      if (status in counts) counts[status]++
    }
    return counts
  }, [runs, effectiveStatus])

  /** Filtered + sorted runs */
  const filteredRuns = useMemo(() => {
    let result = [...runs]

    if (statusFilter !== 'all') {
      result = result.filter((r) => {
        const status = normalizeStatus(effectiveStatus.get(r.name) ?? r.status)
        return status === statusFilter
      })
    }

    if (sortBy === 'recent') {
      result.sort((a, b) => {
        const aRunning = runningNames.has(a.name) ? 1 : 0
        const bRunning = runningNames.has(b.name) ? 1 : 0
        if (aRunning !== bRunning) return bRunning - aRunning
        const aTime = a.created_at ? new Date(a.created_at).getTime() : 0
        const bTime = b.created_at ? new Date(b.created_at).getTime() : 0
        return bTime - aTime
      })
    } else {
      result.sort((a, b) => a.name.localeCompare(b.name))
    }

    return result
  }, [runs, statusFilter, sortBy, effectiveStatus, runningNames])

  // Auto-select the running training run on first load
  useEffect(() => {
    if (autoSelectedRef.current || runningNames.size === 0 || runs.length === 0) return
    const runningRun = runs.find((r) => runningNames.has(r.name))
    if (runningRun) {
      setSelectedRun(runningRun.name)
      autoSelectedRef.current = true
    }
  }, [runningNames, runs])

  // When no run is selected and none exist, default to the new form
  const showNewForm = selectedRun === NEW_RUN
  const currentRun =
    !showNewForm && selectedRun && runs.some((r) => r.name === selectedRun)
      ? selectedRun
      : showNewForm
        ? null
        : (filteredRuns[0]?.name ?? null)

  const isCurrentRunning = runningNames.has(currentRun ?? '')
  const currentStatus = statusRuns.find((r) => r.name === currentRun)
  const gpuBusy = runningNames.size > 0

  // -----------------------------------------------------------------------
  // Render
  // -----------------------------------------------------------------------

  return (
    <>
      <div className="flex h-full flex-col overflow-hidden">
        {/* ============================================================= */}
        {/* Filter bar (below the global page header)                      */}
        {/* ============================================================= */}
        <div className="flex items-center gap-3 border-b border-border px-4 py-1.5">
          {/* + New Run button */}
          <Button
            type="button"
            size="sm"
            variant="outline"
            className={`h-7 shrink-0 gap-1 px-2 text-xs border-primary/50 text-primary hover:bg-primary/10 ${
              showNewForm ? 'bg-primary/10' : ''
            }`}
            onClick={() => setSelectedRun(NEW_RUN)}
          >
            <Plus className="h-3 w-3" />
            New Run
          </Button>

          {/* Filter tabs */}
          {runs.length > 0 && (
            <div className="flex flex-1 gap-1">
              {FILTER_TABS.map((tab) => {
                const count = statusCounts[tab.key] ?? 0
                const isActive = statusFilter === tab.key
                return (
                  <button
                    key={tab.key}
                    type="button"
                    onClick={() => setStatusFilter(tab.key)}
                    className={`rounded-md px-2 py-1 text-[11px] transition-colors ${
                      isActive
                        ? 'bg-primary/10 font-medium text-primary'
                        : 'text-muted-foreground hover:bg-secondary/40 hover:text-foreground'
                    }`}
                  >
                    {tab.label}
                    {count > 0 && tab.key !== 'all' && (
                      <span className="ml-1 opacity-60">{count}</span>
                    )}
                  </button>
                )
              })}
            </div>
          )}
          {runs.length === 0 && <div className="flex-1" />}

          {/* Sort */}
          {runs.length > 0 && (
            <div className="flex items-center gap-1">
              <button
                type="button"
                className={`rounded px-2 py-1 text-[10px] transition-colors ${
                  sortBy === 'recent'
                    ? 'bg-secondary/60 text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setSortBy('recent')}
              >
                Recent
              </button>
              <button
                type="button"
                className={`rounded px-2 py-1 text-[10px] transition-colors ${
                  sortBy === 'name'
                    ? 'bg-secondary/60 text-foreground'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
                onClick={() => setSortBy('name')}
              >
                A–Z
              </button>
            </div>
          )}
        </div>

        {/* ============================================================= */}
        {/* Content: sidebar + detail                                       */}
        {/* ============================================================= */}
        <div className="flex flex-1 overflow-hidden">

          {/* ----- Run List Sidebar ----- */}
          <div className="flex w-56 shrink-0 flex-col border-r border-border/60">
            <div className="flex-1 overflow-y-auto py-1">
              {/* Queued items */}
              {trainingQueue.length > 0 && (
                <div className="border-b border-border/40 pb-1 mb-1">
                  <p className="px-3 pt-1 pb-1 text-[10px] font-semibold uppercase tracking-widest text-muted-foreground/60">
                    Queue
                  </p>
                  {trainingQueue.map((item: TrainingQueueItem) => (
                    <div
                      key={item.id}
                      className="group flex items-center gap-2 px-3 py-1.5"
                    >
                      <Clock className="h-3 w-3 shrink-0 text-muted-foreground/50" />
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-xs text-muted-foreground">{item.name}</div>
                      </div>
                      <span className="shrink-0 rounded-full border border-border px-1.5 py-0.5 text-[9px] text-muted-foreground/60">
                        #{item.position}
                      </span>
                      <button
                        type="button"
                        className="shrink-0 opacity-0 transition-opacity group-hover:opacity-100 text-muted-foreground hover:text-red-400"
                        title="Cancel"
                        onClick={async () => {
                          try {
                            await api.removeFromTrainingQueue(item.id)
                            void queryClient.invalidateQueries({ queryKey: ['training-queue'] })
                          } catch (err) {
                            console.error('Failed to cancel queue item:', err)
                          }
                        }}
                      >
                        <X className="h-3 w-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {filteredRuns.map((run) => {
                const status = effectiveStatus.get(run.name) ?? run.status
                const si = getStatusInfo(status)
                const isActive = !showNewForm && run.name === currentRun
                const isItemRunning = runningNames.has(run.name)
                const sr = statusRuns.find((r) => r.name === run.name)
                const pct =
                  isItemRunning && sr?.total_steps && sr?.current_step
                    ? Math.round((sr.current_step / sr.total_steps) * 100)
                    : null

                return (
                  <button
                    key={run.name}
                    type="button"
                    onClick={() => setSelectedRun(run.name)}
                    className={`group flex w-full flex-col gap-0.5 px-3 py-2 text-left transition-colors ${
                      isActive
                        ? 'bg-primary/10'
                        : 'hover:bg-accent'
                    }`}
                  >
                    {/* Row 1: dot + name + pct */}
                    <div className="flex items-center gap-2">
                      <span
                        className={`h-2 w-2 shrink-0 rounded-full ${si.dot} ${
                          isItemRunning ? 'animate-pulse' : ''
                        }`}
                        title={si.label}
                      />
                      <span
                        className={`min-w-0 flex-1 truncate text-sm ${
                          isActive ? 'font-medium text-primary' : 'text-foreground'
                        }`}
                      >
                        {run.name}
                      </span>
                      {pct !== null && (
                        <span className="shrink-0 text-[10px] font-medium tabular-nums text-emerald-400">
                          {pct}%
                        </span>
                      )}
                    </div>
                    {/* Row 2: base model + trigger word + time ago */}
                    <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                      {(run.base_model || sr?.arch) && (
                        <span className="truncate">{displayModelName(run.base_model ?? sr?.arch)}</span>
                      )}
                      {(run.trigger_word || sr?.trigger_word) && (
                        <>
                          <span className="text-border">&middot;</span>
                          <span className="shrink-0 font-mono text-muted-foreground/70">
                            {run.trigger_word || sr?.trigger_word}
                          </span>
                        </>
                      )}
                      {run.created_at && (
                        <>
                          <span className="text-border">&middot;</span>
                          <span className="shrink-0">{timeAgo(run.created_at)}</span>
                        </>
                      )}
                    </div>
                  </button>
                )
              })}

              {/* Empty list states */}
              {!runsLoading && filteredRuns.length === 0 && !runsError && (
                runs.length === 0 ? (
                  <div className="px-4 py-8 text-center">
                    <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-secondary/40">
                      <Zap className="h-5 w-5 text-muted-foreground" />
                    </div>
                    <p className="text-sm font-medium text-foreground">No training runs yet</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Start your first training run
                    </p>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      className="mx-auto mt-3 gap-1.5 border-primary/50 text-primary hover:bg-primary/10"
                      onClick={() => setSelectedRun(NEW_RUN)}
                    >
                      <Plus className="h-3 w-3" />
                      New Run
                    </Button>
                    <div className="mx-auto mt-2 flex items-center gap-1.5 rounded-md bg-secondary/40 px-3 py-2">
                      <Terminal className="h-3 w-3 shrink-0 text-muted-foreground" />
                      <code className="text-[11px] text-muted-foreground">or use: modl train</code>
                    </div>
                  </div>
                ) : (
                  <div className="px-4 py-8 text-center">
                    <p className="text-sm text-muted-foreground">
                      No {statusFilter} runs
                    </p>
                    <button
                      type="button"
                      className="mt-2 text-xs text-primary hover:underline"
                      onClick={() => setStatusFilter('all')}
                    >
                      Clear filter
                    </button>
                  </div>
                )
              )}
              {runsError && (
                <p className="px-4 py-3 text-xs text-destructive">Failed to load runs.</p>
              )}
            </div>
          </div>

          {/* ----- Detail Area ----- */}
          <div className="flex flex-1 flex-col overflow-hidden">
            {showNewForm ? (
              <NewTrainingForm
                gpuBusy={gpuBusy}
                onStarted={(runName) => {
                  // After starting, select the new run in sidebar
                  setSelectedRun(runName)
                }}
              />
            ) : currentRun ? (
              <TrainingRunCard
                runName={currentRun}
                isRunning={isCurrentRunning}
                gpuBusy={gpuBusy}
                currentStatus={currentStatus}
                onSetLightboxImage={setLightboxImage}
                onSelectRun={setSelectedRun}
              />
            ) : (
              <div className="flex flex-1 flex-col items-center justify-center gap-2 text-center">
                {runs.length > 0 ? (
                  <>
                    <SlidersHorizontal className="h-8 w-8 text-muted-foreground/30" />
                    <p className="text-sm text-muted-foreground">
                      Select a training run to view details
                    </p>
                  </>
                ) : (
                  <div className="max-w-sm space-y-3">
                    <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-full bg-secondary/30">
                      <Zap className="h-6 w-6 text-muted-foreground/40" />
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Train your first LoRA to see results here
                    </p>
                    <div className="mx-auto flex items-center gap-1.5 rounded-md bg-secondary/30 px-3 py-2">
                      <Terminal className="h-3.5 w-3.5 text-muted-foreground" />
                      <code className="text-xs text-primary">
                        modl train --base flux-schnell --dataset your-dataset --lora-type character
                      </code>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sample image lightbox */}
      <SampleLightbox
        image={lightboxImage}
        onClose={() => setLightboxImage(null)}
      />
    </>
  )
}
