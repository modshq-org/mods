import { useState, useMemo } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { BookmarkPlus, Check, ChevronDown } from 'lucide-react'
import { api, type TrainingRun, type PromoteLoraRequest } from '../../api'
import { formatBytes } from '@/lib/utils'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PromoteItem = {
  label: string
  step?: number
  path: string
  size: number
  isFinal: boolean
  promoted: boolean
  sampleUrl?: string
}

export type PromotionDialogProps = {
  runDetail: TrainingRun
  triggerWord?: string
  baseModelRaw?: string
  currentRunKey: string | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

export function buildPromoteItems(runDetail: TrainingRun): PromoteItem[] {
  const items: PromoteItem[] = []

  // Build a map of step -> sample images from the samples array
  const samplesByStep = new Map<number, string>()
  for (const sg of runDetail.samples ?? []) {
    if (sg.images.length > 0) {
      samplesByStep.set(sg.step, sg.images[0])
    }
  }

  // Find closest sample for a given step (highest sample step <= checkpoint step)
  const findSampleForStep = (step: number): string | undefined => {
    let best: string | undefined
    let bestStep = -1
    for (const [sStep, url] of samplesByStep) {
      if (sStep <= step && sStep > bestStep) {
        bestStep = sStep
        best = url
      }
    }
    return best
  }

  // Intermediate checkpoints
  for (const cp of runDetail.checkpoints ?? []) {
    items.push({
      label: `Step ${cp.step.toLocaleString()}`,
      step: cp.step,
      path: cp.path,
      size: cp.size_bytes,
      isFinal: false,
      promoted: cp.promoted,
      sampleUrl: findSampleForStep(cp.step),
    })
  }

  // Final LoRA
  if (runDetail.lora_path) {
    items.push({
      label: 'Final',
      path: runDetail.lora_path,
      size: runDetail.lora_size ?? 0,
      isFinal: true,
      promoted: runDetail.lora_promoted ?? false,
      sampleUrl: samplesByStep.size > 0 ? [...samplesByStep.values()].pop() : undefined,
    })
  }

  return items
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function PromotionDialog({
  runDetail,
  triggerWord,
  baseModelRaw,
  currentRunKey,
}: PromotionDialogProps) {
  const queryClient = useQueryClient()
  const [promoting, setPromoting] = useState(false)
  const [promoted, setPromoted] = useState<string | null>(null)
  const [promoteOpen, setPromoteOpen] = useState(false)

  const promoteItems = useMemo(() => buildPromoteItems(runDetail), [runDetail])

  const handlePromote = async (loraPath: string, step?: number) => {
    setPromoting(true)
    setPromoteOpen(false)
    try {
      const req: PromoteLoraRequest = {
        name: step != null ? `${runDetail.name}-step${step}` : runDetail.name,
        trigger_word: triggerWord ?? undefined,
        base_model: baseModelRaw ?? undefined,
        lora_path: loraPath,
        step,
        training_run: runDetail.name,
        config_json: runDetail.config ? JSON.stringify(runDetail.config) : undefined,
      }
      await api.promoteLora(req)
      setPromoted(runDetail.name)
      void queryClient.invalidateQueries({ queryKey: ['library-loras'] })
      void queryClient.invalidateQueries({ queryKey: ['models'] })
      void queryClient.invalidateQueries({ queryKey: ['run', currentRunKey] })
    } catch (err) {
      console.error('Promote failed:', err)
    } finally {
      setPromoting(false)
    }
  }

  if (promoteItems.length === 0) return null

  if (promoteItems.length === 1) {
    /* Single item: simple button */
    return (
      <Button
        type="button"
        size="sm"
        variant="outline"
        className={`h-7 gap-1.5 px-2 text-xs ${
          promoted === runDetail.name
            ? 'border-emerald-500/50 text-emerald-300'
            : 'border-primary/50 text-primary hover:bg-primary/10'
        }`}
        disabled={promoting || promoted === runDetail.name}
        title="Save this LoRA to your library"
        onClick={() => {
          const item = promoteItems[0]
          void handlePromote(item.path, item.step)
        }}
      >
        {promoted === runDetail.name ? (
          <Check className="h-3 w-3" />
        ) : (
          <BookmarkPlus className="h-3 w-3" />
        )}
        {promoted === runDetail.name
          ? 'Saved'
          : promoting
            ? 'Saving...'
            : 'Save to Library'}
      </Button>
    )
  }

  /* Multiple items: dropdown */
  return (
    <div className="relative">
      <Button
        type="button"
        size="sm"
        variant="outline"
        className={`h-7 gap-1.5 px-2 text-xs ${
          promoted === runDetail.name
            ? 'border-emerald-500/50 text-emerald-300'
            : 'border-primary/50 text-primary hover:bg-primary/10'
        }`}
        disabled={promoting || promoted === runDetail.name}
        onClick={() => setPromoteOpen(!promoteOpen)}
      >
        {promoted === runDetail.name ? (
          <Check className="h-3 w-3" />
        ) : (
          <BookmarkPlus className="h-3 w-3" />
        )}
        {promoted === runDetail.name
          ? 'Saved'
          : promoting
            ? 'Saving...'
            : 'Save to Library'}
        <ChevronDown className="h-3 w-3 opacity-60" />
      </Button>
      {promoteOpen && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setPromoteOpen(false)}
          />
          <div className="absolute left-0 top-full z-50 mt-1 min-w-[280px] rounded-md border border-border bg-card py-1 shadow-lg">
            {promoteItems.map((item, idx) => (
              <button
                key={idx}
                type="button"
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs ${
                  item.promoted
                    ? 'opacity-60'
                    : 'hover:bg-secondary/40'
                }`}
                disabled={item.promoted}
                onClick={() => void handlePromote(item.path, item.step)}
              >
                {item.sampleUrl && (
                  <img
                    src={`/files/${item.sampleUrl}`}
                    alt=""
                    className="size-8 shrink-0 rounded object-cover"
                  />
                )}
                <span className={`flex-1 ${item.isFinal ? 'font-medium text-foreground' : 'text-foreground'}`}>
                  {item.isFinal ? 'Final checkpoint' : item.label}
                </span>
                {item.promoted ? (
                  <span className="shrink-0 rounded bg-emerald-500/10 px-1.5 py-0.5 text-[10px] text-emerald-400">
                    saved
                  </span>
                ) : (
                  <span className="shrink-0 text-muted-foreground">
                    {formatBytes(item.size)}
                  </span>
                )}
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
