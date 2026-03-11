# Inpainting UX

## Backend Status

Already built. CLI supports `--init-image` + `--mask` flags. The worker handles
pipeline switching for compatible architectures (Flux 1 dev/schnell, SDXL, SD1.5).
Mask is a black/white PNG — white = regenerate, black = keep.

## Design: Inline Mask Painting on the Canvas

Inpainting is **not a mode** (no 3rd tab in Generate|Edit). It's an **action on
an existing image**, like upscale or remove-bg. You see an image, want to fix a
region, paint over it, describe the fix, generate.

### Entry Points

1. **Generate canvas toolbar** — "Inpaint" button (brush icon) appears when an
   image is shown. Same row as Download, Edit, Fit.
2. **Image detail modal** — "Inpaint" button in the bottom toolbar alongside
   Recipe, Edit, 4x, BG.
3. **Keyboard shortcut** — `P` (paint) when viewing an image.

All entry points do the same thing: enter paint mode on that image.

### Paint Mode

When activated, the canvas enters an overlay painting state:

```
┌─────────────────────────────────────┐
│  [Brush ●20px] [Eraser] [Clear] [✕] │  ← paint toolbar (top of canvas)
│                                     │
│         ┌───────────────┐           │
│         │               │           │
│         │   image with  │           │
│         │   red/magenta │           │
│         │   mask overlay│           │
│         │               │           │
│         └───────────────┘           │
│                                     │
│  [Generate Inpaint]                 │  ← replaces normal Generate button
└─────────────────────────────────────┘
```

**Painting mechanics:**
- Cursor becomes a circle (brush preview) sized to current brush width
- **Scroll wheel** adjusts brush size (16px–256px)
- **Left-click drag** paints the mask (red/magenta semi-transparent overlay)
- **Right-click drag** or eraser mode erases mask
- **Clear** resets entire mask
- **Esc** or **✕** exits paint mode without generating

**Visual:**
- Masked regions: semi-transparent red/magenta overlay (~40% opacity)
- Unmasked regions: original image visible
- Brush cursor: circle outline matching brush size

### Prompt

The left sidebar prompt field stays active during paint mode. The user types
what should fill the masked area. The prompt could be pre-filled with the
original generation's prompt (if available from metadata).

### Generation

Clicking "Generate Inpaint" (or Ctrl+Enter):

1. Render the mask as a black/white PNG (white = painted/regenerate)
2. Upload mask via `POST /api/upload`
3. Upload init image via `POST /api/upload` (if not already on server)
4. Submit generation with `init_image` + `mask` + `prompt` + `strength`
5. Job enters the normal queue
6. Result appears on canvas like any generation

The `strength` (denoising) parameter controls how much the masked region
diverges from the original. Expose it as a slider in the paint toolbar
(default 0.8, range 0.5–1.0).

### Implementation

**Canvas overlay:** HTML `<canvas>` element positioned absolutely over the
image `<img>`. The canvas stores the mask bitmap. On paint, draw filled circles
at cursor position. On submit, export canvas as PNG blob.

**Key components:**
- `InpaintOverlay.tsx` — the `<canvas>` overlay + paint event handlers
- `PaintToolbar.tsx` — brush size, eraser toggle, clear, strength slider, cancel

**State:**
```typescript
type InpaintState = {
  active: boolean
  sourceImage: PreviewImage | null  // the image being inpainted
  brushSize: number                 // pixels
  erasing: boolean
  strength: number                  // 0.5–1.0
  maskCanvas: HTMLCanvasElement | null
}
```

**Mask rendering:**
```typescript
// Export mask as PNG blob for upload
function exportMask(canvas: HTMLCanvasElement): Blob {
  // Create a new canvas with same dimensions
  // For each pixel: if alpha > 0 → white, else → black
  // Return as PNG blob
}
```

### Model Compatibility

Not all models support inpainting. The "Inpaint" button should be disabled
(with tooltip) when the selected model doesn't support it.

Compatible (from model_family.rs `capabilities.inpaint`):
- flux-dev, flux-schnell
- sdxl
- sd-1.5

Not compatible:
- flux2-* (no inpaint pipeline in diffusers yet)
- z-image, z-image-turbo
- chroma
- qwen-image, qwen-image-edit

### Smart Masks: Segmentation-Assisted Selection

The CLI already has `modl segment` (SAM-based segmentation via BiRefNet/SAM).
Instead of always painting by hand, offer **segmentation overlays** as a
starting point for masks.

#### Flow

1. User enters paint mode on an image
2. Clicks **"Segment"** button in the paint toolbar
3. Server runs segmentation → returns segment polygons/masks
4. Segments render as hoverable colored regions over the image
5. **Click a segment** → adds it to the mask (turns red/magenta like painted areas)
6. Click again → removes it from mask
7. User can then refine with brush/eraser on top

#### Mask Operations

Once segments (or painted regions) are selected, offer simple operations:

| Operation      | What it does                                            |
|----------------|---------------------------------------------------------|
| **Invert**     | Swap masked/unmasked — regenerate everything *except* selection |
| **Expand**     | Dilate mask by N pixels (feathered) — covers seams      |
| **Contract**   | Erode mask by N pixels — tighter selection               |
| **Feather**    | Gaussian blur on mask edges — smoother blending          |

These are pixel operations on the mask canvas, not GPU work. Fast and local.

**Invert** is particularly powerful: select the subject via segmentation, invert,
and you're inpainting the background while keeping the subject untouched.

**Expand** (dilation) is critical for quality: inpainting models need context
around the mask boundary. A mask that's too tight produces visible seams.
Default: auto-expand by 8–16px with soft edges.

#### Segmentation API

Already exists as `POST /api/analysis/segment` (or could be added alongside
upscale/remove-bg). Returns segment masks as:

```json
{
  "segments": [
    { "label": "person", "mask_url": "/files/tmp/seg_0.png", "bbox": [x,y,w,h] },
    { "label": "background", "mask_url": "/files/tmp/seg_1.png", "bbox": [x,y,w,h] }
  ]
}
```

Each `mask_url` is a binary PNG (white = segment, black = not). The frontend
composites these onto the paint canvas when the user clicks a segment.

#### How Inpainting Models Use Context

Key considerations for mask design:

- **Diffusers inpaint pipelines** (Flux, SDXL, SD1.5) receive three inputs:
  `image` (full original), `mask_image` (B/W), and `prompt`
- The model sees the **entire image** as context, not just the masked region.
  The prompt describes the *full scene*, not just the fill content. This means
  writing "a person on a beach" works better than "sand and waves" when
  inpainting the background of a beach photo.
- **Mask feathering** matters: hard mask edges → visible seams. The pipeline
  does some internal blending, but soft mask edges (from Expand/Feather)
  produce significantly better results.
- **Strength** controls how much the masked region can deviate from the original
  pixel values. At strength=1.0 the model fully regenerates; at 0.5 it stays
  close to the original colors/structure. For small fixes (remove object),
  0.75–0.85 works well. For creative replacement, 0.9–1.0.
- **Mask size matters for coherence**: very small masks (<5% of image) may
  produce artifacts because the model has limited room to be coherent. Very
  large masks (>80%) are essentially img2img with extra steps. The sweet spot
  is 10–60% of the image area.
- **Resolution**: the mask must match the image dimensions exactly. The export
  function should resize the canvas mask to match the source image's actual
  pixel dimensions (not the display size).

### Future: Outpainting

The mask canvas could be larger than the source image, with the image centered.
Painting outside the image bounds = outpainting (extend the image). This is a
natural extension of the same mask+prompt mechanism, but needs the worker to
handle padding/compositing. Not in v1.
