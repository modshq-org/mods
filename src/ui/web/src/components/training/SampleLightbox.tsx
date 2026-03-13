import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type SampleLightboxImage = {
  src: string
  step: number
  promptIndex: number
  prompt: string
  runName: string
}

export type SampleLightboxProps = {
  image: SampleLightboxImage | null
  onClose: () => void
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function SampleLightbox({ image, onClose }: SampleLightboxProps) {
  return (
    <Dialog
      open={Boolean(image)}
      onOpenChange={(open) => !open && onClose()}
    >
      <DialogContent className="max-w-[96vw] gap-0 p-0 sm:max-w-4xl">
        {image ? (
          <div className="grid max-h-[90vh] gap-0 lg:grid-cols-[minmax(0,1fr)_280px]">
            <div className="flex items-center justify-center overflow-auto bg-black/70 p-4">
              <img
                src={image.src}
                alt={`Step ${image.step} — Prompt ${image.promptIndex + 1}`}
                className="max-h-[80vh] rounded object-contain"
              />
            </div>
            <div className="flex flex-col border-l border-border/70 bg-card">
              <DialogHeader className="px-4 pt-4">
                <DialogTitle className="text-sm">Sample Details</DialogTitle>
                <DialogDescription className="text-xs">
                  {image.runName}
                </DialogDescription>
              </DialogHeader>
              <div className="flex-1 overflow-auto px-4 py-3">
                <div className="grid grid-cols-[80px_minmax(0,1fr)] gap-x-3 gap-y-3 text-xs">
                  <div className="text-muted-foreground">Step</div>
                  <div className="font-mono text-foreground">
                    {image.step.toLocaleString()}
                  </div>
                  <div className="text-muted-foreground">Prompt&nbsp;#</div>
                  <div className="font-mono text-foreground">
                    {image.promptIndex + 1}
                  </div>
                  <div className="text-muted-foreground">Prompt</div>
                  <div className="break-words leading-relaxed text-foreground">
                    {image.prompt}
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
