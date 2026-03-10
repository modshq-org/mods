import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import type { InstalledModel, ModelFamily } from '../../api'
import { modelDefaults, findModelFamily, type GenerateFormState } from './generate-state'

type Props = {
  models: InstalledModel[]
  families: ModelFamily[]
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  /** When true, auto-adjust steps/guidance on model change */
  autoDefaults?: boolean
}

/** Quality/speed dots (1-5) */
function RatingDots({ value, max = 5 }: { value: number; max?: number }) {
  return (
    <span className="inline-flex gap-0.5">
      {Array.from({ length: max }, (_, i) => (
        <span
          key={i}
          className={`inline-block size-1.5 rounded-full ${
            i < value ? 'bg-primary/80' : 'bg-muted-foreground/20'
          }`}
        />
      ))}
    </span>
  )
}

export function ModelPanel({ models, families, form, setForm, autoDefaults = true }: Props) {
  const checkpoints = models.filter((m) => m.model_type === 'checkpoint' || m.model_type === 'diffusion_model')

  // Find family info for the selected model
  const selectedModel = checkpoints.find((m) => m.id === form.base_model_id)
  const familyInfo = selectedModel ? findModelFamily(selectedModel.name, families) : null

  const handleChange = (modelId: string) => {
    const model = checkpoints.find((m) => m.id === modelId)
    if (!model) return

    setForm((prev) => {
      const patch: Partial<GenerateFormState> = { base_model_id: modelId }

      if (autoDefaults) {
        const info = findModelFamily(model.name, families)
        const defaults = modelDefaults(model.name, info)
        patch.steps = defaults.steps
        patch.guidance = defaults.guidance
      }

      return { ...prev, ...patch }
    })
  }

  return (
    <div className="space-y-1.5">
      <span className="text-[11px] font-medium text-muted-foreground">
        Checkpoint
      </span>
      <Select
        value={form.base_model_id}
        onValueChange={handleChange}
        disabled={checkpoints.length === 0}
      >
        <SelectTrigger className="w-full bg-background/60">
          <SelectValue placeholder={checkpoints.length === 0 ? 'No checkpoints installed' : 'Select a model'} />
        </SelectTrigger>
        <SelectContent>
          {checkpoints.map((model) => {
            const info = findModelFamily(model.name, families)
            return (
              <SelectItem key={model.id} value={model.id}>
                <div className="flex items-center gap-2">
                  <span>{model.name}</span>
                  {model.variant && (
                    <span className="text-[10px] text-muted-foreground">({model.variant})</span>
                  )}
                  {info && (
                    <span className="text-[10px] text-muted-foreground/60">
                      {info.total_b}B
                    </span>
                  )}
                  <span className="ml-auto text-[10px] text-muted-foreground/60">
                    {(model.size_bytes / 1024 / 1024 / 1024).toFixed(1)}GB
                  </span>
                </div>
              </SelectItem>
            )
          })}
        </SelectContent>
      </Select>

      {/* Model info bar */}
      {familyInfo && (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md bg-secondary/20 px-2 py-1.5">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-muted-foreground/60">quality</span>
            <RatingDots value={familyInfo.quality} />
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] text-muted-foreground/60">speed</span>
            <RatingDots value={familyInfo.speed} />
          </div>
          {familyInfo.text_rendering && (
            <span className="rounded bg-primary/10 px-1 py-0.5 text-[9px] font-medium text-primary">
              text
            </span>
          )}
          {/* Capability badges */}
          <div className="flex gap-1">
            {familyInfo.capabilities.img2img && (
              <span className="rounded bg-secondary/60 px-1 py-0.5 text-[9px] text-muted-foreground">
                img2img
              </span>
            )}
            {familyInfo.capabilities.inpaint && (
              <span className="rounded bg-secondary/60 px-1 py-0.5 text-[9px] text-muted-foreground">
                inpaint
              </span>
            )}
            {familyInfo.capabilities.edit && (
              <span className="rounded bg-secondary/60 px-1 py-0.5 text-[9px] text-muted-foreground">
                edit
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
