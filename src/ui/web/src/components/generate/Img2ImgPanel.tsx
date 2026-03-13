import { useCallback, useRef } from 'react'
import { AlertTriangleIcon, ImageIcon, UploadIcon, XIcon } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import type { ModelFamilyInfo } from '../../api'
import type { GenerateFormState } from './generate-state'

type Props = {
  form: GenerateFormState
  setForm: React.Dispatch<React.SetStateAction<GenerateFormState>>
  /** Family info for the currently selected model (null if unknown) */
  modelInfo: ModelFamilyInfo | null
}

export function Img2ImgPanel({ form, setForm, modelInfo }: Props) {
  const fileRef = useRef<HTMLInputElement>(null)
  const supportsImg2img = modelInfo?.capabilities.img2img ?? true

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) return
      const reader = new FileReader()
      reader.onloadend = () => {
        setForm((prev) => ({
          ...prev,
          mode: 'img2img',
          init_image: reader.result as string,
          init_image_file: file,
        }))
      }
      reader.readAsDataURL(file)
    },
    [setForm],
  )

  const clearImage = () => {
    setForm((prev) => ({
      ...prev,
      mode: 'txt2img',
      init_image: null,
      init_image_file: null,
    }))
    if (fileRef.current) fileRef.current.value = ''
  }

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      const file = e.dataTransfer.files[0]
      handleFile(file)
    },
    [handleFile],
  )

  return (
    <div className="space-y-2.5">
      {/* Warning when model doesn't support img2img */}
      {!supportsImg2img && (
        <div className="flex items-start gap-2 rounded-md border border-yellow-500/20 bg-yellow-500/5 px-2.5 py-2">
          <AlertTriangleIcon className="mt-0.5 size-3 flex-shrink-0 text-yellow-500/70" />
          <span className="text-[10px] text-yellow-500/80">
            {modelInfo?.name ?? 'This model'} does not support img2img.
            Use Flux Dev, Flux Schnell, SDXL, or SD 1.5.
          </span>
        </div>
      )}

      {/* Drop zone / preview */}
      {form.init_image ? (
        <div className="relative">
          <img
            src={form.init_image}
            alt="Init image"
            className="h-32 w-full rounded-md border border-border/60 object-contain bg-secondary/20"
          />
          <Button
            type="button"
            variant="ghost"
            size="icon"
            className="absolute right-1 top-1 size-6 bg-background/80 backdrop-blur"
            onClick={clearImage}
          >
            <XIcon className="size-3" />
          </Button>
        </div>
      ) : (
        <div
          className={`flex h-24 cursor-pointer flex-col items-center justify-center gap-1.5 rounded-md border border-dashed transition-colors ${
            supportsImg2img
              ? 'border-border/60 bg-secondary/10 hover:border-primary/40 hover:bg-secondary/20'
              : 'border-border/30 bg-secondary/5 opacity-50'
          }`}
          onClick={() => supportsImg2img && fileRef.current?.click()}
          onDragOver={(e) => e.preventDefault()}
          onDrop={(e) => supportsImg2img && handleDrop(e)}
        >
          <UploadIcon className="size-5 text-muted-foreground/40" />
          <span className="text-[10px] text-muted-foreground/50">
            {supportsImg2img ? 'Drop image or click to upload' : 'Not supported by this model'}
          </span>
        </div>
      )}

      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files?.[0])}
      />

      {/* Denoise strength (only when image is set) */}
      {form.init_image && (
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground/70">Denoise strength</span>
            <span className="font-mono text-[10px] text-muted-foreground">
              {form.denoise_strength.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[form.denoise_strength]}
            min={0}
            max={1}
            step={0.05}
            onValueChange={([v]) =>
              setForm((prev) => ({ ...prev, denoise_strength: v ?? prev.denoise_strength }))
            }
          />
          <p className="text-[10px] text-muted-foreground/40">
            0 = no change, 1 = fully regenerate
          </p>
        </div>
      )}

      {!form.init_image && supportsImg2img && (
        <p className="flex items-center gap-1.5 text-[10px] text-muted-foreground/40">
          <ImageIcon className="size-3" />
          Upload an image to use img2img mode
        </p>
      )}
    </div>
  )
}
