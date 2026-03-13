import { useState, useMemo } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { CollapsibleSection } from '@/components/ui/collapsible-section'
import {
  Database,
  Loader2,
  Sparkles,
  User,
  Palette,
  Box,
} from 'lucide-react'
import { toast } from 'sonner'
import { api, type DatasetSummary, type InstalledModel, type StartTrainingRequest } from '../../api'
import { STALE_MODERATE, STALE_FAST } from '@/lib/query-keys'

type LoraType = 'character' | 'style' | 'object'

const LORA_TYPES: { key: LoraType; label: string; description: string; icon: React.ElementType; defaultTrigger: string }[] = [
  { key: 'character', label: 'Character', description: 'A specific person, character, or face', icon: User, defaultTrigger: 'OHWX' },
  { key: 'style', label: 'Style', description: 'An artistic style, aesthetic, or look', icon: Palette, defaultTrigger: 'STYLETOK' },
  { key: 'object', label: 'Object', description: 'A specific object, product, or concept', icon: Box, defaultTrigger: 'OBJTOK' },
]

const PRESETS: { key: string; label: string; description: string }[] = [
  { key: 'quick', label: 'Quick', description: 'Fast training, good for testing (fewer steps)' },
  { key: 'standard', label: 'Standard', description: 'Recommended balance of quality and speed' },
  { key: 'advanced', label: 'Advanced', description: 'Full control — configure all parameters below' },
]

type Props = {
  gpuBusy: boolean
  onStarted: (runName: string) => void
}

export function NewTrainingForm({ gpuBusy, onStarted }: Props) {
  const queryClient = useQueryClient()
  const [starting, setStarting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Form state
  const [selectedDataset, setSelectedDataset] = useState<string | null>(null)
  const [loraType, setLoraType] = useState<LoraType>('character')
  const [triggerWord, setTriggerWord] = useState('OHWX')
  const [selectedModel, setSelectedModel] = useState<string | null>(null)
  const [preset, setPreset] = useState('standard')
  const [name, setName] = useState('')

  // Advanced params
  const [advSteps, setAdvSteps] = useState('')
  const [advRank, setAdvRank] = useState('')
  const [advLr, setAdvLr] = useState('')
  const [advOptimizer, setAdvOptimizer] = useState('adamw8bit')
  const [advSeed, setAdvSeed] = useState('')
  const [advClassWord, setAdvClassWord] = useState('')

  // Queries
  const { data: datasets = [], isLoading: datasetsLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: api.datasets,
    staleTime: STALE_MODERATE,
  })

  const { data: modelsResponse } = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    staleTime: STALE_FAST,
  })

  // Filter to training-capable models
  const trainableModels = useMemo(() => {
    const models = modelsResponse?.models ?? []
    return models.filter(
      (m) =>
        (m.model_type === 'checkpoint' || m.model_type === 'diffusion_model') &&
        !m.id.startsWith('train:'),
    )
  }, [modelsResponse])

  // Auto-generate name
  const autoName = useMemo(() => {
    if (!selectedDataset) return ''
    return `${selectedDataset}-v1`
  }, [selectedDataset])

  const effectiveName = name || autoName

  const canStart = selectedDataset && selectedModel && effectiveName.trim() && loraType && triggerWord.trim()

  const handleStart = async () => {
    if (!canStart) return
    setStarting(true)
    setError(null)
    try {
      const req: StartTrainingRequest = {
        dataset: selectedDataset!,
        base_model: selectedModel!,
        name: effectiveName.trim(),
        trigger_word: triggerWord.trim(),
        lora_type: loraType,
        preset: preset !== 'advanced' ? preset : undefined,
        steps: advSteps && !isNaN(parseInt(advSteps, 10)) ? parseInt(advSteps, 10) : undefined,
        rank: advRank && !isNaN(parseInt(advRank, 10)) ? parseInt(advRank, 10) : undefined,
        lr: advLr && !isNaN(parseFloat(advLr)) ? parseFloat(advLr) : undefined,
        optimizer: advOptimizer !== 'adamw8bit' ? advOptimizer : undefined,
        seed: advSeed && !isNaN(parseInt(advSeed, 10)) ? parseInt(advSeed, 10) : undefined,
        class_word: advClassWord.trim() || undefined,
      }
      if (gpuBusy) {
        await api.addToTrainingQueue(req)
        void queryClient.invalidateQueries({ queryKey: ['training-queue'] })
        toast.success(`"${effectiveName}" added to training queue`)
      } else {
        await api.startTraining(req)
        void queryClient.invalidateQueries({ queryKey: ['runs'] })
        void queryClient.invalidateQueries({ queryKey: ['status'] })
        toast.success(`Training started: ${effectiveName}`)
      }
      onStarted(effectiveName.trim())
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to start training')
    } finally {
      setStarting(false)
    }
  }

  const isAdvanced = preset === 'advanced'

  return (
    <div className="flex h-full flex-col">
      {/* Scrollable form */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-2xl px-6 py-6 space-y-0">
          {/* Title */}
          <div className="mb-6">
            <h2 className="flex items-center gap-2 text-lg font-semibold text-foreground">
              <Sparkles className="h-5 w-5 text-primary" />
              New Training Run
            </h2>
            <p className="mt-1 text-xs text-muted-foreground">
              Train a LoRA adapter on your dataset
            </p>
          </div>

          {/* ── Dataset ── */}
          <CollapsibleSection title="Dataset" defaultOpen={true}>
            {datasetsLoading ? (
              <div className="py-4 text-center text-xs text-muted-foreground">Loading datasets...</div>
            ) : datasets.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border/60 px-4 py-6 text-center">
                <Database className="mx-auto mb-2 h-6 w-6 text-muted-foreground/40" />
                <p className="text-xs text-muted-foreground">
                  No datasets found. Create one with <code className="rounded bg-secondary px-1 py-0.5 text-primary">modl dataset create</code>
                </p>
              </div>
            ) : (
              <div className="grid gap-2 sm:grid-cols-2">
                {datasets.map((ds: DatasetSummary) => (
                  <button
                    key={ds.name}
                    type="button"
                    onClick={() => setSelectedDataset(ds.name)}
                    className={`flex items-center gap-3 rounded-lg border px-3 py-2.5 text-left transition-colors ${
                      selectedDataset === ds.name
                        ? 'border-primary bg-primary/5'
                        : 'border-border/60 hover:border-border hover:bg-secondary/20'
                    }`}
                  >
                    <Database className="h-4 w-4 shrink-0 text-muted-foreground" />
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-foreground">{ds.name}</div>
                      <div className="text-[10px] text-muted-foreground">
                        {ds.image_count > 0 ? (
                          <>
                            {ds.image_count} images
                            {ds.coverage > 0 && (
                              <span className="ml-1">{Math.round(ds.coverage * 100)}% captioned</span>
                            )}
                          </>
                        ) : (
                          'Dataset'
                        )}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </CollapsibleSection>

          {/* ── Concept ── */}
          <CollapsibleSection title="Concept" defaultOpen={true}>
            <div className="space-y-4">
              {/* LoRA type */}
              <div>
                <label className="mb-2 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  LoRA Type
                </label>
                <div className="grid gap-2 sm:grid-cols-3">
                  {LORA_TYPES.map((lt) => {
                    const Icon = lt.icon
                    return (
                      <button
                        key={lt.key}
                        type="button"
                        onClick={() => {
                          setLoraType(lt.key)
                          setTriggerWord(lt.defaultTrigger)
                        }}
                        className={`flex flex-col items-center gap-1.5 rounded-lg border px-3 py-3 text-center transition-colors ${
                          loraType === lt.key
                            ? 'border-primary bg-primary/5'
                            : 'border-border/60 hover:border-border hover:bg-secondary/20'
                        }`}
                      >
                        <Icon className={`h-5 w-5 ${loraType === lt.key ? 'text-primary' : 'text-muted-foreground'}`} />
                        <div className="text-xs font-medium text-foreground">{lt.label}</div>
                        <div className="text-[10px] leading-tight text-muted-foreground">{lt.description}</div>
                      </button>
                    )
                  })}
                </div>
              </div>

              {/* Trigger word */}
              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  Trigger Word
                </label>
                <input
                  type="text"
                  value={triggerWord}
                  onChange={(e) => setTriggerWord(e.target.value)}
                  placeholder="e.g. OHWX"
                  className="h-9 w-full rounded-md border border-border bg-secondary/30 px-3 text-sm text-foreground placeholder:text-muted-foreground/60 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
                />
                <p className="mt-1 text-[10px] text-muted-foreground">
                  A unique token to activate the LoRA in prompts
                </p>
              </div>
            </div>
          </CollapsibleSection>

          {/* ── Base Model ── */}
          <CollapsibleSection title="Base Model" defaultOpen={true}>
            {trainableModels.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border/60 px-4 py-6 text-center">
                <p className="text-xs text-muted-foreground">
                  No trainable models installed. Install one with <code className="rounded bg-secondary px-1 py-0.5 text-primary">modl pull flux-schnell</code>
                </p>
              </div>
            ) : (
              <div className="space-y-1.5 max-h-48 overflow-y-auto">
                {trainableModels.map((m: InstalledModel) => (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setSelectedModel(m.id)}
                    className={`flex w-full items-center gap-3 rounded-lg border px-3 py-2 text-left transition-colors ${
                      selectedModel === m.id
                        ? 'border-primary bg-primary/5'
                        : 'border-border/60 hover:border-border hover:bg-secondary/20'
                    }`}
                  >
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-foreground">{m.name}</div>
                      <div className="text-[10px] text-muted-foreground">
                        {m.variant && <span>{m.variant} · </span>}
                        {m.id}
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </CollapsibleSection>

          {/* ── Training Preset ── */}
          <CollapsibleSection title="Training Preset" defaultOpen={true}>
            <div className="grid gap-2 sm:grid-cols-3">
              {PRESETS.map((p) => (
                <button
                  key={p.key}
                  type="button"
                  onClick={() => setPreset(p.key)}
                  className={`flex flex-col rounded-lg border px-3 py-2.5 text-left transition-colors ${
                    preset === p.key
                      ? 'border-primary bg-primary/5'
                      : 'border-border/60 hover:border-border hover:bg-secondary/20'
                  }`}
                >
                  <div className="text-sm font-medium text-foreground">{p.label}</div>
                  <div className="text-[10px] text-muted-foreground">{p.description}</div>
                </button>
              ))}
            </div>
          </CollapsibleSection>

          {/* ── Run Name ── */}
          <CollapsibleSection title="Run Name" defaultOpen={true}>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={autoName || 'my-lora-v1'}
              className="h-9 w-full rounded-md border border-border bg-secondary/30 px-3 text-sm text-foreground placeholder:text-muted-foreground/60 focus:border-primary/50 focus:outline-none focus:ring-1 focus:ring-primary/30"
            />
          </CollapsibleSection>

          {/* ── Advanced (collapsed by default) ── */}
          <CollapsibleSection title="Advanced" defaultOpen={isAdvanced}>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
              <div>
                <label className="mb-1 block text-[10px] text-muted-foreground">Steps</label>
                <input
                  type="number"
                  value={advSteps}
                  onChange={(e) => setAdvSteps(e.target.value)}
                  placeholder="auto"
                  className="h-8 w-full rounded border border-border bg-secondary/30 px-2 text-xs text-foreground placeholder:text-muted-foreground/40 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-1 block text-[10px] text-muted-foreground">Rank</label>
                <input
                  type="number"
                  value={advRank}
                  onChange={(e) => setAdvRank(e.target.value)}
                  placeholder="16"
                  className="h-8 w-full rounded border border-border bg-secondary/30 px-2 text-xs text-foreground placeholder:text-muted-foreground/40 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-1 block text-[10px] text-muted-foreground">Learning Rate</label>
                <input
                  type="text"
                  value={advLr}
                  onChange={(e) => setAdvLr(e.target.value)}
                  placeholder="auto"
                  className="h-8 w-full rounded border border-border bg-secondary/30 px-2 text-xs text-foreground placeholder:text-muted-foreground/40 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-1 block text-[10px] text-muted-foreground">Optimizer</label>
                <select
                  value={advOptimizer}
                  onChange={(e) => setAdvOptimizer(e.target.value)}
                  className="h-8 w-full rounded border border-border bg-secondary/30 px-2 text-xs text-foreground focus:outline-none"
                >
                  <option value="adamw8bit">adamw8bit</option>
                  <option value="prodigy">prodigy</option>
                </select>
              </div>
              <div>
                <label className="mb-1 block text-[10px] text-muted-foreground">Seed</label>
                <input
                  type="number"
                  value={advSeed}
                  onChange={(e) => setAdvSeed(e.target.value)}
                  placeholder="random"
                  className="h-8 w-full rounded border border-border bg-secondary/30 px-2 text-xs text-foreground placeholder:text-muted-foreground/40 focus:outline-none"
                />
              </div>
              <div>
                <label className="mb-1 block text-[10px] text-muted-foreground">Class Word</label>
                <input
                  type="text"
                  value={advClassWord}
                  onChange={(e) => setAdvClassWord(e.target.value)}
                  placeholder="auto"
                  className="h-8 w-full rounded border border-border bg-secondary/30 px-2 text-xs text-foreground placeholder:text-muted-foreground/40 focus:outline-none"
                />
              </div>
            </div>
          </CollapsibleSection>
        </div>
      </div>

      {/* Sticky footer */}
      <div className="shrink-0 border-t border-border bg-[#0e0e18]/95 px-6 py-3 backdrop-blur">
        <div className="mx-auto flex max-w-2xl items-center gap-3">
          {gpuBusy && (
            <div className="flex-1 rounded-md border border-amber-500/30 bg-amber-500/5 px-3 py-1.5 text-[11px] text-amber-300">
              GPU is busy — this run will be queued
            </div>
          )}
          {error && (
            <div className="flex-1 rounded-md border border-red-500/30 bg-red-500/5 px-3 py-1.5 text-[11px] text-red-400">
              {error}
            </div>
          )}
          <div className="flex-1" />
          <Button
            type="button"
            size="sm"
            className="gap-1.5"
            disabled={!canStart || starting}
            onClick={() => void handleStart()}
          >
            {starting ? (
              <Loader2 className="h-3 w-3 animate-spin" />
            ) : (
              <Sparkles className="h-3 w-3" />
            )}
            {starting ? 'Starting...' : gpuBusy ? 'Add to Queue' : 'Start Training'}
          </Button>
        </div>
      </div>
    </div>
  )
}
