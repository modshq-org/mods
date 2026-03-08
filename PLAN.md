# PLAN.md — Future Ideas & Roadmap

This document captures planned improvements and feature ideas that go beyond
the current implementation. Items here are aspirational — they may or may not
be built, and the design may change.

---

## LoRA Management & Generation UX

### Implemented

- **Trigger word display** — Trained LoRAs show their trigger word in the
  generate form. Clicking the chip inserts it into the prompt.
- **Sample thumbnails** — Trained LoRAs show a thumbnail from their latest
  training sample in the selector and in the active LoRA card.
- **Base model metadata** — The `/api/models` endpoint returns `trigger_word`,
  `base_model_id`, and `sample_image_url` for LoRAs that have artifact records.

### Future Ideas

- **Intermediate step management** — Training produces many intermediate
  checkpoints (e.g. `art-style-v4_000002035`, `_000004070`, etc.) that get
  installed as separate LoRAs. These clutter the LoRA list. Ideas:
  - Group intermediates under their parent training run in the UI
  - Let users "promote" a specific step to be the primary LoRA
  - Auto-hide intermediates, show only the final or promoted step
  - `modl prune-intermediates <run>` CLI command to remove all but the best

- **LoRA comparison view** — Side-by-side generation with different LoRAs or
  different steps of the same training run. Useful for picking the best
  checkpoint. Could show a grid: rows = steps, columns = different prompts.

- **External LoRA import with metadata** — When importing a LoRA from
  CivitAI or HuggingFace, scrape/store trigger words and sample images.
  The `modl pull` flow could extract this from the registry manifest or
  from model card metadata.

- **Favorite LoRAs** — Pin frequently-used LoRAs to the top of the selector.
  Could reuse the existing `favorites` table in state.db.

- **LoRA compatibility warnings** — Warn when a LoRA's `base_model_id` doesn't
  match the currently selected checkpoint (e.g. SDXL LoRA with SD1.5 base).

- **Trigger word auto-insert** — When a LoRA is added to the form, automatically
  prepend its trigger word to the prompt (with an undo toast). Currently the user
  must click the chip manually.

- **LoRA strength presets** — Save per-LoRA preferred strength values so they
  persist across sessions. Currently defaults to 0.8 every time.

- **Batch LoRA testing** — Generate the same prompt with varying LoRA strengths
  (0.4, 0.6, 0.8, 1.0) in one batch to find the sweet spot.

---

## Generation

- **Flux text encoder fix** — `from_single_file()` doesn't load text encoders
  for Flux models. Needs explicit text encoder loading in the Python executor.
  SDXL works fine; Flux is blocked on this.

- **Generation queue** — Allow queueing multiple generation requests instead of
  blocking on one at a time. The current `AtomicBool` lock prevents concurrent
  runs; a proper job queue would let users stack requests.

- **Generation history search** — Filter/search past generations by prompt,
  model, LoRA, or date. Currently just shows a flat chronological gallery.

- **A/B prompt comparison** — Generate two prompts side-by-side with identical
  settings to compare prompt phrasing.

---

## Training

- **Live loss chart** — The training status already streams loss values. Render
  a sparkline or mini chart in the UI instead of just the current number.

- **Training presets** — Save and load training configurations (learning rate,
  steps, resolution, etc.) as named presets.

- **Resume from UI** — Currently resuming a training run requires the CLI.
  Add a "Resume" button in the training detail view.

---

## Image Primitives — Composable Operations for the Agent Loop

The goal: expose every useful image operation as a standalone CLI command and
worker action. An LLM agent composes these primitives into arbitrary workflows
— no node graph, no manual wiring. Each primitive has a clear input/output
contract, runs through the persistent worker for GPU caching, and uses
auto-managed utility models.

**Design principles:**
- **One command = one operation.** `modl segment`, not `modl perceive --mode segment`.
- **Uniform I/O.** Images in, images + JSON out. Masks are grayscale PNGs.
  Scores are JSON. Everything serializable so the agent can chain outputs.
- **Auto-pull models.** Utility models (<500MB) auto-download on first use
  into `~/.modl/models/utility/`. Larger adapters (ControlNet, IP-Adapter)
  use `modl pull` like any other model.
- **Worker-native.** Each primitive is a worker action (`{"action": "segment", ...}`)
  alongside the existing `"generate"`. Models stay in VRAM across calls.
- **CLI + API + Agent.** Every primitive is a CLI command, an HTTP API endpoint,
  and an agent tool — same as existing generate/train.

### Perception — Extract information from images

#### `modl segment` — Segment Anything

Extract a mask from an image using a text prompt, bounding box, or point.

```
modl segment photo.jpg --prompt "the person"           → mask.png
modl segment photo.jpg --bbox 100,50,400,300           → mask.png
modl segment photo.jpg --point 250,175                 → mask.png
modl segment photo.jpg --prompt "face" --expand 20%    → mask.png (dilated)
modl segment photo.jpg --prompt "background" --invert  → mask.png
```

- **Models:** SAM 2.1 (Meta, ~375MB) + GroundingDINO (for text prompts, ~900MB)
- **Output:** Grayscale mask PNG (white = selected region, black = excluded)
- **Worker action:** `{"action": "segment", "image": "...", "prompt": "...", "mode": "text|bbox|point"}`
- **Why:** Foundation for inpainting, face fixing, background replacement, compositing.
  The agent calls this before almost every targeted edit.

#### `modl detect` — Object & Face Detection

Find objects or faces in an image. Returns bounding boxes and optional crops.

```
modl detect photo.jpg --type face                      → JSON [{bbox, confidence, landmarks}]
modl detect photo.jpg --type face --crop               → face_0.png, face_1.png + JSON
modl detect photo.jpg --prompt "red car"               → JSON [{bbox, confidence, label}]
modl detect photo.jpg --type face --embeddings         → JSON [{bbox, embedding: [512 floats]}]
```

- **Face detection:** InsightFace buffalo_l (~1.2GB, also gives 512-d face embeddings)
- **Open-vocab detection:** GroundingDINO (shared with segment, ~900MB)
- **Output:** JSON array of detections. `--crop` also saves cropped regions.
  `--embeddings` includes the face recognition vector (for compare/swap).
- **Why:** The agent needs to know _what's in an image_ before deciding what to
  do. Face detection is prerequisite for face-fix, face-swap, face-comparison.
  Object detection enables targeted inpainting ("replace the chair").

#### `modl extract` — Condition Maps

Generate structural condition maps from an image (depth, pose, edges, normals).

```
modl extract photo.jpg --type depth                    → depth.png
modl extract photo.jpg --type pose                     → pose.png + JSON [{keypoints}]
modl extract photo.jpg --type edge                     → edge.png
modl extract photo.jpg --type normal                   → normal.png
```

- **Depth:** DepthAnything v2 (~700MB). Best general-purpose monocular depth.
- **Pose:** DWPose / RTMPose (~200MB). 2D body + hand + face keypoints.
- **Edge:** Canny (pure OpenCV, no model) or PiDiNet (~50MB) for soft edges.
- **Normal:** Omnidata or Marigold-derived (~700MB). Surface normal estimation.
- **Output:** Condition map as PNG + optional JSON metadata (keypoints for pose).
- **Why:** These are inputs to ControlNet generation. The agent calls
  `modl extract --type pose` → gets pose map → feeds it to
  `modl generate --controlnet pose --control-image pose.png`.

#### `modl score` — Image Quality & Relevance Scoring

Rate an image on quality, aesthetics, or prompt adherence.

```
modl score photo.jpg                                   → {"aesthetic": 6.8}
modl score photo.jpg --metric aesthetic                → {"aesthetic": 6.8, "percentile": 85}
modl score photo.jpg --metric clip --prompt "a dog"    → {"clip_score": 0.31}
modl score photo.jpg --metric all --prompt "a dog"     → {"aesthetic": 6.8, "clip_score": 0.31}
```

- **Aesthetic:** LAION aesthetic predictor v2 (~28MB on top of CLIP).
  Scores 1-10, trained on human preferences.
- **CLIP-text:** OpenCLIP ViT-L/14 (~1.7GB, shared with other CLIP uses).
  Cosine similarity between image and text embeddings.
- **Output:** JSON with scores.
- **Why:** The agent needs a feedback signal. After generating, it calls
  `modl score` to decide if the result is good enough or needs retry.
  Critical for automatic checkpoint evaluation during training.

#### `modl compare` — Image Similarity

Compare two images across different dimensions.

```
modl compare a.png b.png                               → {"clip": 0.92, "lpips": 0.08}
modl compare a.png b.png --metric face                 → {"face_similarity": 0.87}
modl compare a.png b.png --metric clip                 → {"clip_similarity": 0.92}
modl compare a.png b.png --metric lpips                → {"lpips_distance": 0.08}
modl compare a.png b.png --metric all                  → {all metrics}
```

- **CLIP similarity:** Same CLIP model as score. Compares semantic content.
- **Face similarity:** InsightFace embeddings (shared with detect). Cosine
  distance between face vectors. Essential for character LoRA evaluation.
- **LPIPS:** Learned Perceptual Image Patch Similarity (~50MB).
  Lower = more similar. Good for checking if inpainting preserved surroundings.
- **Output:** JSON with similarity scores.
- **Why:** The agent compares original vs. result to verify edits preserved
  what they should. Compares training samples vs. generations to evaluate
  LoRA fidelity. Compares faces across images for identity consistency.

### Generation Extensions — Beyond txt2img

#### `modl inpaint` — Masked Region Regeneration

Regenerate a specific region of an image guided by a mask and prompt.

```
modl inpaint photo.jpg --mask mask.png --prompt "blue eyes"
modl inpaint photo.jpg --mask mask.png --prompt "wooden floor" --strength 0.9
modl inpaint photo.jpg --mask mask.png --prompt "sharp detailed face" --base flux-dev --padding 32
```

- **Implementation:** Uses the same diffusion pipeline with mask conditioning.
  For Flux: `FluxInpaintPipeline`. For SDXL: `StableDiffusionXLInpaintPipeline`.
  The mask defines the regeneration region; `--padding` adds context around it.
- **Strength:** 0.0 = keep original, 1.0 = fully regenerate masked area.
- **Output:** Full image with masked region replaced. Same dimensions as input.
- **Agent pattern:** `segment → inpaint` is the core edit loop. The agent
  segments a face, dilates the mask, inpaints with "sharp detailed face".
  This is what FaceDetailer / ADetailer does — but composable.

#### `modl generate --controlnet` — Structural Guidance

Generate images guided by a structural condition (depth, pose, edge, etc.).

```
modl generate "fashion model" --controlnet depth --control-image depth.png
modl generate "same pose, different outfit" --controlnet pose --control-image pose.png
modl generate "architectural sketch" --controlnet edge --control-image edges.png --control-strength 0.8
```

- **Models:** ControlNet adapters per condition type (~1.5GB each). Pulled
  explicitly: `modl pull controlnet-depth-flux`, `modl pull controlnet-pose-flux`.
- **`--control-strength`:** 0.0-1.0, how strictly to follow the condition.
- **Multiple controls:** `--controlnet depth,pose --control-image d.png,p.png`
  for multi-ControlNet (less common but powerful).
- **Agent pattern:** `extract depth → generate with controlnet-depth` gives
  the agent precise structural control over the output.

#### `modl generate --reference` — IP-Adapter / Reference Image Guidance

Generate guided by a reference image for style or face consistency.

```
modl generate "portrait in a garden" --reference face.jpg --reference-type face
modl generate "product on marble" --reference style-ref.jpg --reference-type style
modl generate "same person, different scene" --reference photo.jpg --reference-type face --reference-strength 0.7
```

- **Models:** IP-Adapter (~1.5GB) + CLIP vision encoder (~1.7GB).
  For face: IP-Adapter-FaceID uses InsightFace embeddings instead of CLIP.
  Pulled via `modl pull ip-adapter-face-flux`.
- **`--reference-type face`:** Uses InsightFace embedding → IP-Adapter-FaceID.
  Preserves identity without training a LoRA. Good for one-shot face consistency.
- **`--reference-type style`:** Uses CLIP image embedding → standard IP-Adapter.
  Transfers artistic style from a reference image.
- **Agent pattern:** For quick one-off face consistency (no training needed),
  the agent uses `--reference-type face`. For deeper style transfer, it trains
  a LoRA. The agent picks the right approach based on the task.

### Transforms — Modify Images Without Diffusion

#### `modl upscale` — Super Resolution

Upscale an image 2x or 4x using a dedicated upscaler model.

```
modl upscale photo.jpg                                 → photo_4x.png
modl upscale photo.jpg --scale 2                       → photo_2x.png
modl upscale photo.jpg --model realesrgan              → photo_4x.png
modl upscale photo.jpg --face                          → photo_4x.png (face-enhanced)
```

- **Models:** Real-ESRGAN 4x (~64MB), Real-ESRGAN 4x-anime (~64MB).
  `--face` adds GFPGAN face enhancement during upscale.
- **Output:** Upscaled PNG at 2x or 4x the input resolution.
- **Agent pattern:** Generate at 1024px → upscale 4x to 4096px.
  Cheaper and faster than generating at 4096px directly.

#### `modl face-restore` — Face Enhancement

Fix distorted, blurry, or artifacted faces in generated images.

```
modl face-restore photo.jpg                            → photo_restored.png
modl face-restore photo.jpg --model codeformer         → photo_restored.png
modl face-restore photo.jpg --strength 0.7             → photo_restored.png (subtle)
```

- **Models:** CodeFormer (~375MB, best quality), GFPGAN v1.4 (~350MB, faster).
- **`--strength`:** 0.0 = keep original face, 1.0 = full restoration.
  Lower values preserve more of the original while fixing artifacts.
- **Output:** Full image with all detected faces restored.
- **Agent pattern:** After generation, if `modl score` reports low aesthetic
  and `modl detect --type face` found faces, the agent tries face-restore
  before resorting to the heavier segment → inpaint loop.

#### `modl remove-bg` — Background Removal

Remove the background from an image, producing a transparent PNG.

```
modl remove-bg photo.jpg                               → photo_nobg.png
modl remove-bg photo.jpg --model rmbg2                 → photo_nobg.png
modl remove-bg photo.jpg --mask-only                   → mask.png
```

- **Models:** RMBG 2.0 (BRIA, ~180MB) or BiRefNet (~900MB, higher quality edges).
- **`--mask-only`:** Output just the mask (useful for compositing workflows).
- **Output:** RGBA PNG with transparent background, or grayscale mask.
- **Agent pattern:** Product photo workflow: `remove-bg → generate background →
  composite`. Also useful for training data preparation (isolate subject).

#### `modl face-swap` — Face Transfer

Replace a face in a target image with a face from a source image.

```
modl face-swap --source face.jpg --target scene.jpg    → swapped.png
modl face-swap --source face.jpg --target scene.jpg --face-index 0  → swapped.png
```

- **Models:** InsightFace inswapper (~500MB, shared buffalo_l detector).
- **`--face-index`:** Which face in the target to replace (when multiple faces).
- **Output:** Target image with the specified face replaced.
- **Agent pattern:** Alternative to IP-Adapter-FaceID for putting a specific
  person into a generated scene. Generate scene → face-swap source face in.
  Lower quality than LoRA but zero training time.

#### `modl composite` — Layer Compositing

Combine foreground and background images using a mask.

```
modl composite fg.png bg.png --mask mask.png           → result.png
modl composite fg.png bg.png --position center         → result.png
modl composite fg.png bg.png --mask mask.png --feather 5  → result.png (soft edges)
```

- **No model needed.** Pure image operation (PIL/OpenCV).
- **`--feather`:** Gaussian blur on mask edges for seamless blending.
- **`--position`:** Placement when fg is smaller than bg (center, top-left, etc.).
- **Output:** Composited image.
- **Agent pattern:** The final step in many workflows. Remove-bg → generate
  new background → composite. Or inpaint face → composite back into original
  at higher quality.

### Training Evaluation — Automatic Checkpoint Selection

#### `modl train eval` — Score & Rank LoRA Checkpoints

Generate test images across all checkpoints of a training run, score them,
and rank which checkpoint is best.

```
modl train eval my-lora-run
modl train eval my-lora-run --prompts "photo of OHWX at the beach, photo of OHWX in a studio"
modl train eval my-lora-run --metric aesthetic+face --reference training-photos/
modl train eval my-lora-run --top 3                    → shows top 3 checkpoints
```

- **How it works:**
  1. Lists all intermediate checkpoints for the run (e.g. steps 500, 1000, 1500, 2000)
  2. Generates N test images per checkpoint using provided or auto-generated prompts
  3. Scores each generation: aesthetic score, CLIP-text alignment, face similarity
     (if `--reference` provided with original training photos)
  4. Outputs a ranked table: step | aesthetic | clip | face_sim | overall
  5. Optionally auto-promotes the best checkpoint

- **Metrics:**
  - `aesthetic` — LAION aesthetic score (general quality)
  - `clip` — Prompt adherence (does the image match the prompt?)
  - `face` — Face similarity to reference photos (character LoRAs only)
  - `style` — CLIP-image similarity to training data (style LoRAs)

- **Agent pattern:** After training completes, the agent calls `modl train eval`
  to find the optimal checkpoint instead of blindly using the final one.
  This replaces the manual "eyeball each sample" workflow.

#### `modl train promote` — Promote a Checkpoint

Mark a specific intermediate step as the "primary" LoRA for a training run.

```
modl train promote my-lora-run --step 2035
modl train promote my-lora-run --best                  → auto-selects from eval results
```

- Moves the selected checkpoint to the primary LoRA name.
- Hides other intermediates from the default `modl ls` output.
- Records the promotion in the DB for provenance.

### Model Management — All Through the Registry

Every model used by primitives gets a manifest in `modl-registry`, same as
checkpoints and LoRAs. This gives us version pinning, SHA256 verification,
VRAM-based variant selection, and shared dependency resolution for free.

#### New asset type: `utility`

```yaml
# modl-registry/manifests/utility/sam2.yaml
id: sam2
name: "SAM 2.1 — Segment Anything Model 2"
type: utility
architecture: sam2
capability: [segmentation]
author: meta
license: apache-2.0
homepage: https://github.com/facebookresearch/sam2

variants:
  - id: tiny
    file: sam2.1_hiera_tiny.pt
    url: https://huggingface.co/facebook/sam2.1-hiera-tiny/resolve/main/sam2.1_hiera_tiny.pt
    sha256: "..."
    size: 40000000
    vram_required: 512
  - id: base
    file: sam2.1_hiera_base_plus.pt
    url: https://huggingface.co/facebook/sam2.1-hiera-base-plus/resolve/main/sam2.1_hiera_base_plus.pt
    sha256: "..."
    size: 375000000
    vram_required: 1024
    default: true
  - id: large
    file: sam2.1_hiera_large.pt
    url: https://huggingface.co/facebook/sam2.1-hiera-large/resolve/main/sam2.1_hiera_large.pt
    sha256: "..."
    size: 2400000000
    vram_required: 4096

tags: [segmentation, masks, perception]
```

The `capability` field is the key addition. When `modl segment` runs, it
resolves the model like:
1. Check if any model with `capability: segmentation` is installed
2. If not, auto-pull the default variant (VRAM-aware, same as checkpoints)
3. Cache in the worker alongside diffusion pipelines

Users can override: `modl segment photo.jpg --model sam2-large` or even
swap to a different segmentation model entirely if one gets added to the
registry later (e.g. Florence-2, a future SAM 3).

#### Registry manifests needed

These all go in `modl-registry/manifests/utility/`:

**Segmentation & Detection:**

| Manifest ID | Model | HuggingFace Source | Variants | Capability |
|------------|-------|-------------------|----------|------------|
| `sam2` | SAM 2.1 | `facebook/sam2.1-hiera-*` | tiny/base/large | segmentation |
| `grounding-dino` | GroundingDINO 1.5 | `IDEA-Research/grounding-dino-*` | base/large | detection, grounding |
| `insightface` | InsightFace buffalo_l | `deepinsight/insightface` | buffalo_l | face-detection, face-embedding |

**Condition Extraction:**

| Manifest ID | Model | HuggingFace Source | Variants | Capability |
|------------|-------|-------------------|----------|------------|
| `depth-anything` | DepthAnything v2 | `depth-anything/Depth-Anything-V2-*` | small/base/large | depth-estimation |
| `dwpose` | DWPose (RTMPose) | `yzd-v/DWPose` | base | pose-estimation |
| `pidinet` | PiDiNet | `lllyasviel/Annotators` | base | edge-detection |

**Scoring & Comparison:**

| Manifest ID | Model | HuggingFace Source | Variants | Capability |
|------------|-------|-------------------|----------|------------|
| `clip-vit-l` | OpenCLIP ViT-L/14 | `laion/CLIP-ViT-L-14-*` | fp16/fp32 | clip-embedding |
| `aesthetic-predictor` | LAION Aesthetic v2 | `christophschuhmann/improved-aesthetic-predictor` | base | aesthetic-scoring |
| `lpips` | LPIPS (AlexNet) | `richzhang/PerceptualSimilarity` | base | perceptual-similarity |

**Transforms:**

| Manifest ID | Model | HuggingFace Source | Variants | Capability |
|------------|-------|-------------------|----------|------------|
| `realesrgan` | Real-ESRGAN | `ai-forever/Real-ESRGAN` | 4x/4x-anime | upscaling |
| `codeformer` | CodeFormer | `sczhou/CodeFormer` | base | face-restoration |
| `gfpgan` | GFPGAN v1.4 | `TencentARC/GFPGAN` | v1.4 | face-restoration |
| `rmbg2` | RMBG 2.0 | `briaai/RMBG-2.0` | base | background-removal |
| `birefnet` | BiRefNet | `ZhengPeng7/BiRefNet` | base | background-removal |
| `inswapper` | InsightFace inswapper | `deepinsight/insightface` | 128 | face-swap |

**ControlNet adapters** (explicit `modl pull`, not auto-pull):

| Manifest ID | Model | HuggingFace Source | Size | Capability |
|------------|-------|-------------------|------|------------|
| `controlnet-depth-flux` | Flux ControlNet Depth | `Shakker-Labs/FLUX.1-dev-ControlNet-Depth` | ~1.5GB | controlnet-depth |
| `controlnet-pose-flux` | Flux ControlNet Pose | `Shakker-Labs/FLUX.1-dev-ControlNet-Pose` | ~1.5GB | controlnet-pose |
| `controlnet-canny-flux` | Flux ControlNet Canny | `Shakker-Labs/FLUX.1-dev-ControlNet-Canny` | ~1.5GB | controlnet-edge |
| `controlnet-depth-sdxl` | SDXL ControlNet Depth | `diffusers/controlnet-depth-sdxl-1.0` | ~2.5GB | controlnet-depth |
| `controlnet-canny-sdxl` | SDXL ControlNet Canny | `diffusers/controlnet-canny-sdxl-1.0` | ~2.5GB | controlnet-edge |

**IP-Adapter** (explicit `modl pull`):

| Manifest ID | Model | HuggingFace Source | Size | Capability |
|------------|-------|-------------------|------|------------|
| `ip-adapter-flux` | IP-Adapter Flux | `InstantX/FLUX.1-dev-IP-Adapter` | ~1.5GB | ip-adapter-style |
| `ip-adapter-face-flux` | IP-Adapter FaceID Flux | `InstantX/FLUX.1-dev-IP-Adapter` | ~1.5GB | ip-adapter-face |
| `clip-vision-h` | CLIP ViT-H/14 (vision) | `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` | ~1.7GB | clip-vision |

#### Capability-based resolution

The `capability` field enables loose coupling between commands and models.
`modl segment` doesn't hardcode "use SAM2" — it asks the registry for any
model with `capability: segmentation`. This means:

- If a user installs `sam2-large` instead of the default `sam2-base`, segment
  uses it automatically.
- If a future model (Florence-2 SAM, etc.) is added with the same capability,
  it becomes a drop-in replacement.
- Multiple models with the same capability = user picks via `--model`, or the
  system uses the most recently installed one.
- The agent can list available capabilities and pick the right model for
  the task.

#### Auto-pull vs explicit pull

- **Auto-pull** (utility models, <~1.5GB): `modl segment` auto-installs
  `sam2` on first use. No manual step. The user sees
  `"Installing SAM 2.1 base (375 MB)..."` on first run, then it's cached.
- **Explicit pull** (ControlNet, IP-Adapter, >~1.5GB): `modl pull controlnet-depth-flux`.
  If the user tries `modl generate --controlnet depth` without it,
  they get a helpful error: `"ControlNet depth adapter not installed. Run: modl pull controlnet-depth-flux"`.

#### Worker model cache expansion

The persistent worker's `ModelCache` currently manages diffusion pipelines
only. It needs a second tier for utility models:

```python
class ModelCache:
    _diffusion_cache: dict[CacheKey, CachedPipeline]   # existing
    _utility_cache: dict[str, Any]                       # new: "sam2" → loaded model
```

Utility models are much smaller than diffusion pipelines (~100MB-1.7GB vs
~12-24GB), so the worker can hold several in VRAM simultaneously. Eviction
priority: utility models stay loaded longer (cheap to keep), diffusion
pipelines evict first (expensive in VRAM).

### Agent Tool Expansion

The current agent has 7 tools (analyze → generate). With primitives, it gets:

```
Perception:    segment, detect, extract, score, compare
Generation:    generate (extended), inpaint
Transforms:    upscale, face_restore, remove_bg, face_swap, composite
Training:      train_lora, eval_checkpoints, promote_checkpoint
Existing:      analyze_images, create_dataset, caption_images,
               select_base_model, enhance_prompt
```

The agent system prompt is updated to describe these as composable building
blocks rather than a fixed linear pipeline. Example agent reasoning:

```
User: "Make this photo look like a watercolor painting but keep my face"
Agent thinks:
  1. detect --type face → get face bbox + embedding
  2. segment --prompt "face" → face_mask.png
  3. generate --reference style-ref.jpg --reference-type style → styled.png
  4. compare original.jpg styled.png --metric face → 0.3 (face changed too much)
  5. inpaint styled.png --mask face_mask.png --prompt "original face, sharp" → result.png
  6. compare original.jpg result.png --metric face → 0.82 (face preserved)
  7. score result.png --metric aesthetic → 7.1 (good quality)
  → Done
```

### Implementation Priority

**Phase 1 — Core edit loop** (highest value, enables face-fix and inpainting):
1. `segment` (SAM2 + GroundingDINO)
2. `inpaint` (FluxInpaintPipeline / SDXL inpaint)
3. `detect --type face` (InsightFace)
4. `score --metric aesthetic` (LAION aesthetic)

**Phase 2 — Transform & evaluate** (quality-of-life, post-processing):
5. `upscale` (Real-ESRGAN)
6. `face-restore` (CodeFormer)
7. `remove-bg` (RMBG 2.0)
8. `compare` (CLIP + face similarity)
9. `train eval` + `train promote`

**Phase 3 — Structural control** (advanced generation):
10. `extract` (depth, pose, edge)
11. `generate --controlnet`
12. `generate --reference` (IP-Adapter)
13. `face-swap` (InsightFace inswapper)
14. `composite`

---

## Workflows — Shareable, Reproducible Image Pipelines

Workflows are the ComfyUI replacement. Instead of a JSON node graph with
hundreds of lines of wiring, a workflow is a short YAML file that chains
modl primitives. Readable, diffable, shareable, versionable.

```
modl run face-fix.yaml --input photo.jpg
modl run product-shoot.yaml --input product.jpg --var bg_prompt="marble countertop"
modl run evaluate-lora.yaml --var run=pedro-v3 --var reference=./headshots/
```

### Workflow Format

A workflow is a YAML file with inputs, steps, and outputs:

```yaml
# face-fix.yaml — Detect and regenerate bad faces at higher detail
name: face-fix
description: ADetailer-equivalent face fix pipeline
version: 1

inputs:
  image: { type: image, required: true }
  prompt: { type: string, default: "sharp detailed face, natural skin" }
  base: { type: string, default: "flux-dev" }

steps:
  - id: find_faces
    run: detect
    with:
      image: ${input.image}
      type: face
      crop: true

  - id: face_mask
    run: segment
    each: ${find_faces.detections}       # loop over each detected face
    with:
      image: ${input.image}
      bbox: ${item.bbox}
      expand: "25%"

  - id: fix_face
    run: inpaint
    each: ${face_mask.results}
    with:
      image: ${input.image}
      mask: ${item.mask}
      prompt: ${input.prompt}
      base: ${input.base}
      strength: 0.65
      padding: 32

  - id: quality_check
    run: score
    with:
      image: ${fix_face.output}
      metric: aesthetic

output:
  image: ${fix_face.output}
  score: ${quality_check.score}
```

### Step Reference & Variable System

- **`${input.X}`** — References a workflow input
- **`${step_id.output}`** — References the primary output of a previous step
- **`${step_id.field}`** — References a specific field from a step's JSON output
- **`${item.X}`** — Inside an `each:` loop, references the current item
- **`${var.X}`** — References a runtime variable from `--var key=value`

Steps execute sequentially by default. Steps with no interdependencies
could run in parallel (future optimization), but sequential is simpler
and matches how most workflows actually flow.

### `each:` — Looping Over Results

The `each:` field on a step makes it execute once per item in a list.
This handles the common case of "do X for every face detected" or
"process every image in a folder":

```yaml
  - id: fix_face
    run: inpaint
    each: ${find_faces.detections}    # detections is an array
    with:
      image: ${input.image}
      mask: ${item.mask}              # item = current detection
```

### `when:` — Conditional Steps

Steps can be conditional based on previous results:

```yaml
  - id: maybe_restore
    run: face-restore
    when: ${quality_check.aesthetic} < 5.0    # only if score is low
    with:
      image: ${fix_face.output}
      strength: 0.5
```

The agent uses these extensively — it generates a workflow with conditionals
baked in based on its reasoning about what might go wrong.

### `retry:` — Quality Gates

A step can retry with different params if a quality check fails:

```yaml
  - id: generate
    run: generate
    with:
      prompt: ${input.prompt}
      base: flux-dev
    retry:
      max: 3
      until: ${self.score.aesthetic} > 6.0
      score:
        run: score
        with:
          image: ${self.output}
          metric: aesthetic
```

### Workflow Lock File — `workflow.lock`

When you run a workflow, modl generates a lock file that pins the exact
models used. This makes the workflow fully reproducible:

```yaml
# workflow.lock — Auto-generated, do not edit
generated: 2026-03-07T14:30:00Z
modl_version: 0.3.0
workflow: face-fix.yaml
workflow_hash: sha256:a1b2c3...

models:
  - id: flux-dev
    type: diffusion_model
    variant: fp8
    sha256: "e5f6g7..."
    source: modl-registry

  - id: sam2
    type: utility
    variant: base
    sha256: "x1y2z3..."
    capability: segmentation
    source: modl-registry

  - id: insightface
    type: utility
    variant: buffalo_l
    sha256: "m3n4o5..."
    capability: face-detection
    source: modl-registry

  - id: aesthetic-predictor
    type: utility
    variant: base
    sha256: "p6q7r8..."
    capability: aesthetic-scoring
    source: modl-registry
```

Running `modl run face-fix.yaml --lock workflow.lock` ensures the exact
same model versions are used. Share the `.yaml` + `.lock` pair for
reproducibility. The lock file is auto-generated on first run and can
be committed alongside the workflow.

### Built-in Workflows — Ship With Modl

Modl ships a curated set of workflows that cover the most common ComfyUI
use cases. These live in the modl-registry (or a `workflows/` dir in it)
and can be pulled: `modl workflow pull face-fix`.

**Core workflows (Phase 1):**

#### `face-fix` — ADetailer / FaceDetailer Replacement
```
detect faces → segment each face → inpaint each face → score result
```
This is the #1 most used ComfyUI workflow. Every portrait generation
benefits from it. One command: `modl run face-fix --input gen.png`.

#### `hires-fix` — Hi-Res Fix / Upscale + Refine
```
generate at 1024px → upscale 2x → img2img at low strength (0.3-0.5)
```
Classic A1111/ComfyUI pattern. Generate fast at low res, upscale, then
refine. Saves VRAM and time vs. generating at 2048px directly.

#### `background-replace` — Background Swap
```
remove-bg → generate new background from prompt → composite with feather
```
Product photography, profile photos, creative compositing. Input: subject
photo + background prompt. Output: subject on new background.

#### `product-photo` — Full Product Photography Pipeline
```
remove-bg → generate scene → composite → upscale → score
```
E-commerce use case. Upload product on white bg, get lifestyle shots.

#### `style-transfer-safe` — Style Transfer Preserving Identity
```
detect face + get embedding → generate with style reference →
compare face similarity → if low: segment face + inpaint original face back
```
The tricky workflow that ComfyUI users spend hours wiring. Here it's a
single file with a quality gate on face preservation.

**Extended workflows (Phase 2):**

#### `pose-transfer` — Recreate a Pose
```
extract pose from reference → generate with controlnet-pose + prompt
```

#### `batch-eval` — Evaluate LoRA Checkpoints
```
for each checkpoint: load LoRA → generate N test images → score →
compare faces to reference → rank → promote best
```

#### `face-swap-scene` — Put Someone in a Generated Scene
```
generate scene → detect face in scene → face-swap source face in →
face-restore → score
```

#### `iterative-refine` — Generate Until Good Enough
```
enhance prompt → generate → score → if aesthetic < threshold:
adjust prompt + retry (up to N times)
```

#### `consistent-batch` — Generate N Images with Face Consistency
```
for each prompt: generate with face reference → compare face to
reference → if similarity < 0.8: face-swap + restore → collect results
```

#### `training-data-prep` — Prepare Dataset from Raw Photos
```
for each image: remove-bg (optional) → resize → auto-caption →
validate quality via score → reject low-quality images
```

### What This Replaces from ComfyUI

| ComfyUI Workflow | Nodes Required | Modl Equivalent | Steps |
|-----------------|----------------|-----------------|-------|
| Basic txt2img | 5-8 | `modl generate` | 1 |
| ADetailer face fix | 12-18 | `modl run face-fix` | 4 |
| Hi-res fix | 8-12 | `modl run hires-fix` | 3 |
| Background replacement | 10-15 | `modl run background-replace` | 3 |
| IP-Adapter style transfer | 8-12 | `modl generate --reference` | 1 |
| IP-Adapter face consistency | 10-15 | `modl generate --reference-type face` | 1 |
| ControlNet pose | 8-12 | `modl extract + modl generate --controlnet` | 2 |
| Face swap (ReActor) | 6-10 | `modl face-swap` | 1 |
| Upscale + face restore | 6-10 | `modl upscale --face` | 1 |
| Product photography | 15-25 | `modl run product-photo` | 5 |
| Batch with quality gate | 20-30 | `modl run iterative-refine` | 3-4 |
| LoRA checkpoint eval | Manual | `modl run batch-eval` | 4 |

That's 80-90% of what people actually build in ComfyUI. The remaining
10-20% is niche stuff (regional prompting, multi-ControlNet stacking,
custom CLIP manipulation, AnimateDiff video, etc.) that stays in ComfyUI
territory — and that's fine. Modl's symlink system already integrates
with ComfyUI for those edge cases.

### `modl run` — The Workflow Executor

```
modl run <workflow.yaml> [--input <image>] [--var key=value ...] [--lock <file>] [--dry-run] [--json]
```

- Parses the workflow YAML
- Resolves all model dependencies (auto-pull utility models, check adapters)
- Generates a lock file if one doesn't exist
- Executes steps sequentially through the persistent worker
- Streams progress via events (CLI progress bars / SSE to web UI)
- Outputs the final result(s)

The web UI gets a "Workflows" page where users can browse built-in workflows,
upload custom ones, fill in inputs via a form, and watch execution in real-time.
This is the visual ComfyUI replacement — but the workflow itself is just a YAML
file you can edit in any text editor.

### Agent + Workflows

The agent can both **use** and **create** workflows:

- **Use:** The agent picks from built-in workflows when they match the user's
  intent. "Fix the faces in this image" → the agent runs `face-fix.yaml`
  rather than manually calling segment + inpaint.

- **Create:** For novel requests, the agent composes a custom workflow on the
  fly from primitives, saves it as a `.yaml` file, and runs it. The user can
  then re-use that workflow later. "Create a workflow that processes my product
  photos" → agent writes `my-product.yaml` and runs it.

This is the key differentiator: ComfyUI users manually build workflows.
Modl users describe what they want, and the agent builds the workflow
automatically. And because workflows are just YAML, the agent's output is
human-readable, editable, and shareable — not an opaque graph.

### Workflow Registry

Workflows can be shared through the same registry infrastructure:

```
modl workflow pull face-fix              # from modl-registry/workflows/
modl workflow pull product-photo
modl workflow ls                         # list installed workflows
modl workflow ls --available             # list registry workflows
```

Community-contributed workflows via PR to the registry, same as model
manifests. Each workflow in the registry includes:
- The YAML file
- A description and preview images
- A recommended lock file (pinned model versions that are tested)
- Tags for discoverability

```
modl-registry/
  manifests/
    checkpoints/...
    loras/...
    utility/              ← NEW: all utility model manifests
      sam2.yaml
      grounding-dino.yaml
      insightface.yaml
      depth-anything.yaml
      dwpose.yaml
      clip-vit-l.yaml
      aesthetic-predictor.yaml
      lpips.yaml
      realesrgan.yaml
      codeformer.yaml
      gfpgan.yaml
      rmbg2.yaml
      birefnet.yaml
      inswapper.yaml
      pidinet.yaml
    controlnet/           ← NEW: controlnet adapter manifests
      controlnet-depth-flux.yaml
      controlnet-pose-flux.yaml
      controlnet-canny-flux.yaml
      controlnet-depth-sdxl.yaml
      controlnet-canny-sdxl.yaml
    ip_adapter/           ← NEW: ip-adapter manifests
      ip-adapter-flux.yaml
      ip-adapter-face-flux.yaml
      clip-vision-h.yaml
  workflows/              ← NEW: shareable workflow recipes
    face-fix.yaml
    hires-fix.yaml
    background-replace.yaml
    product-photo.yaml
    style-transfer-safe.yaml
    pose-transfer.yaml
    batch-eval.yaml
    face-swap-scene.yaml
    iterative-refine.yaml
    consistent-batch.yaml
    training-data-prep.yaml
  index.json

---

## Infrastructure

- **Daemonized training** — Run training as a background daemon instead of a
  child process, so it survives terminal disconnects without tmux/screen.

- **Multi-GPU support** — Detect and offer GPU selection when multiple GPUs
  are available.

- **Remote generation** — Send generation requests to a remote GPU (Modal,
  RunPod) when local GPU is unavailable or too slow.
