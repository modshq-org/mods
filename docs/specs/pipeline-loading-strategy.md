# Pipeline Loading Strategy

**Status:** Implementation spec
**Date:** 2026-03-09

## Problem

When a user runs `modl generate` with a locally installed model (e.g. `flux-dev`), the
Python worker receives a `base_model_path` pointing to a bare `.safetensors` file in the
modl store (e.g. `~/modl/store/checkpoint/47d8.../flux1-dev-fp8-e4m3fn.safetensors`).

The current `load_pipeline()` detects this is a split-component model and **falls back to
downloading from HuggingFace** via `from_pretrained(hf_repo)` — ignoring:
- The user's locally installed files
- Their chosen quantization (fp8 vs fp16 vs GGUF)
- Potentially OOMing by loading fp16 weights on a 24GB card

## Goal

Load pipelines from the modl store using whatever components the user has installed,
respecting their quantization choices. No HuggingFace downloads at inference time.

---

## Supported Model Families

### Tier 1 — Build & test now (models user has installed)

| Model ID | Pipeline Class | Transformer Class | Text Encoder | VAE | Notes |
|---|---|---|---|---|---|
| `flux-dev` | `FluxPipeline` | `FluxTransformer2DModel` | CLIP-L + T5-XXL | `AutoencoderKL` | fp8 transformer installed |
| `flux-schnell` | `FluxPipeline` | `FluxTransformer2DModel` | CLIP-L + T5-XXL | `AutoencoderKL` | needs training adapter |
| `z-image-turbo` | `ZImagePipeline` | `ZImageTransformer2DModel` | Qwen3 4B | `AutoencoderKL` | Already works (from_pretrained HF) |
| `z-image` | `ZImagePipeline` | `ZImageTransformer2DModel` | Qwen3 4B | `AutoencoderKL` | |

### Tier 2 — Add arch_config + test when models available

| Model ID | Pipeline Class | Transformer Class | Text Encoder | VAE | Notes |
|---|---|---|---|---|---|
| `qwen-image` | `QwenImagePipeline` | `QwenImageTransformer2DModel` | `Qwen2_5_VLForConditionalGeneration` | `AutoencoderKLQwenImage` | 20B model, GGUF variants in registry |
| `qwen-image-edit` | `QwenImageEditPipeline` | `QwenImageTransformer2DModel` | `Qwen2_5_VLForConditionalGeneration` | `AutoencoderKLQwenImage` | Has safetensors variants |
| `flux2-dev` | `Flux2Pipeline` | `Flux2Transformer2DModel` | `Mistral3ForConditionalGeneration` | `AutoencoderKLFlux2` | New architecture, gated |
| `flux2-klein-4b` | `Flux2KleinPipeline` | `Flux2Transformer2DModel` | `Qwen3ForCausalLM` | `AutoencoderKLFlux2` | 4B params, 4-step distilled |
| `flux2-klein-9b` | `Flux2KleinPipeline` | `Flux2Transformer2DModel` | `Qwen3ForCausalLM` | `AutoencoderKLFlux2` | |
| `chroma` | `ChromaPipeline` | `ChromaTransformer2DModel` | T5-XXL | `AutoencoderKL` | Flux-based VAE |

### Tier 3 — Already working (full checkpoints)

| Model ID | Pipeline Class | Loading Method | Notes |
|---|---|---|---|
| `sdxl-base-1.0` | `StableDiffusionXLPipeline` | `from_single_file()` | No changes needed |
| `sdxl-turbo` | `StableDiffusionXLPipeline` | `from_single_file()` | |
| `sd-1.5` | `StableDiffusionPipeline` | `from_single_file()` | |

---

## Component Map

Config files are **per-component**, shared across model families. Each config dir contains
the JSON/tokenizer files diffusers needs to know the architecture of a bare safetensors.

### Flux 1 Components

```
Pipeline: FluxPipeline(transformer, text_encoder, tokenizer, text_encoder_2, tokenizer_2, vae, scheduler)
```

| Pipeline Param | Diffusers Class | modl Store ID | Config Dir |
|---|---|---|---|
| transformer | `FluxTransformer2DModel` | (base model) | `flux-dev-transformer/` or `flux-schnell-transformer/` |
| text_encoder | `CLIPTextModel` | `clip-l` | `clip-l/` |
| tokenizer | `CLIPTokenizer` | — (config only) | `clip-tokenizer/` |
| text_encoder_2 | `T5EncoderModel` | `t5-xxl-fp8` or `t5-xxl-fp16` | `t5-xxl/` |
| tokenizer_2 | `T5TokenizerFast` | — (config only) | `t5-tokenizer/` |
| vae | `AutoencoderKL` | `flux-vae` | `flux-vae/` |
| scheduler | `FlowMatchEulerDiscreteScheduler` | — (config only) | `flux-dev-scheduler/` or `flux-schnell-scheduler/` |

### Z-Image / Z-Image-Turbo Components

```
Pipeline: ZImagePipeline(transformer, text_encoder, tokenizer, vae, scheduler)
```

| Pipeline Param | Diffusers Class | modl Store ID | Config Dir |
|---|---|---|---|
| transformer | `ZImageTransformer2DModel` | (base model) | `zimage-transformer/` |
| text_encoder | `Qwen3Model` | `z-image-text-encoder` | `qwen3-text-encoder/` |
| tokenizer | `Qwen2Tokenizer` | — (config only) | `qwen2-tokenizer/` |
| vae | `AutoencoderKL` | `z-image-vae` | `zimage-vae/` |
| scheduler | `FlowMatchEulerDiscreteScheduler` | — (config only) | `zimage-scheduler/` |

### Qwen-Image Components

```
Pipeline: QwenImagePipeline(transformer, text_encoder, tokenizer, vae, scheduler)
```

| Pipeline Param | Diffusers Class | modl Store ID | Config Dir |
|---|---|---|---|
| transformer | `QwenImageTransformer2DModel` | (base model) | `qwen-image-transformer/` |
| text_encoder | `Qwen2_5_VLForConditionalGeneration` | `qwen-image-clip` | `qwen-image-text-encoder/` |
| tokenizer | `Qwen2Tokenizer` | — (config only) | `qwen-image-tokenizer/` |
| vae | `AutoencoderKLQwenImage` | `qwen-image-vae` | `qwen-image-vae/` |
| scheduler | `FlowMatchEulerDiscreteScheduler` | — (config only) | `qwen-image-scheduler/` |

Note: QwenImage uses a **special VAE class** (`AutoencoderKLQwenImage`), not the standard `AutoencoderKL`.

Edit pipelines also need `processor: Qwen2VLProcessor`.

### Flux 2 Dev Components

```
Pipeline: Flux2Pipeline(transformer, text_encoder, tokenizer, vae, scheduler)
```

| Pipeline Param | Diffusers Class | modl Store ID | Config Dir |
|---|---|---|---|
| transformer | `Flux2Transformer2DModel` | (base model) | `flux2-dev-transformer/` |
| text_encoder | `Mistral3ForConditionalGeneration` | `flux2-mistral-text-encoder` | `flux2-mistral-te/` |
| tokenizer | `AutoProcessor` (PixtralProcessor) | — (config only) | `flux2-mistral-tokenizer/` |
| vae | `AutoencoderKLFlux2` | `flux2-vae` | `flux2-vae/` |
| scheduler | `FlowMatchEulerDiscreteScheduler` | — (config only) | `flux2-dev-scheduler/` |

### Flux 2 Klein Components

```
Pipeline: Flux2KleinPipeline(transformer, text_encoder, tokenizer, vae, scheduler)
```

| Pipeline Param | Diffusers Class | modl Store ID | Config Dir |
|---|---|---|---|
| transformer | `Flux2Transformer2DModel` | (base model) | `flux2-klein-transformer/` |
| text_encoder | `Qwen3ForCausalLM` | `flux2-qwen3-4b-text-encoder` | `flux2-qwen3-te/` |
| tokenizer | `Qwen2TokenizerFast` | — (config only) | `flux2-qwen3-tokenizer/` |
| vae | `AutoencoderKLFlux2` | `flux2-vae` | `flux2-vae/` (shared with Dev) |
| scheduler | `FlowMatchEulerDiscreteScheduler` | — (config only) | `flux2-klein-scheduler/` |

### Chroma Components

```
Pipeline: ChromaPipeline(transformer, text_encoder, tokenizer, vae, scheduler)
```

| Pipeline Param | Diffusers Class | modl Store ID | Config Dir |
|---|---|---|---|
| transformer | `ChromaTransformer2DModel` | (base model) | `chroma-transformer/` |
| text_encoder | `T5EncoderModel` | `t5-xxl-fp8` or `t5-xxl-fp16` | `t5-xxl/` (shared with Flux) |
| tokenizer | `T5Tokenizer` | — (config only) | `chroma-tokenizer/` |
| vae | `AutoencoderKL` | `flux-vae` | `flux-vae/` (shared with Flux) |
| scheduler | `FlowMatchEulerDiscreteScheduler` | — (config only) | `chroma-scheduler/` |

**Bug fix needed:** Current arch_config says Chroma uses `z-image-text-encoder` (Qwen3).
It actually uses T5 (same as Flux). Must be corrected.

---

## GGUF Support

Diffusers supports loading GGUF via `from_single_file()` with `GGUFQuantizationConfig`:

```python
from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig

transformer = FluxTransformer2DModel.from_single_file(
    "path/to/model-Q4_K_M.gguf",
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    config="black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
)
```

This means **all GGUF variants in the registry** can potentially be loaded:
- Qwen-Image GGUF (Q2_K through Q8_0)
- Flux 2 GGUF (Q2_K through BF16)
- Flux 2 Klein GGUF variants

For GGUF loading, we need:
1. Detect `.gguf` extension in `detect_model_format()`
2. Use `TransformerClass.from_single_file(path, quantization_config=GGUFQuantizationConfig(...))`
3. Still need bundled config for the transformer (via `config=` param)
4. Assembly of other components (TE, VAE, tokenizer) is the same as safetensors

---

## Architecture

### Three Loading Strategies

```
detect_model_format(model_source) → format
  ├─ "hf_directory"      → PipelineClass.from_pretrained(dir)
  ├─ "full_checkpoint"   → PipelineClass.from_single_file(path)
  ├─ "gguf"              → assemble_pipeline() with GGUF transformer
  └─ "transformer_only"  → assemble_pipeline() from local components
```

### `detect_model_format(path: str) -> str`

1. If directory containing `model_index.json` → `"hf_directory"`
2. If `.gguf` extension → `"gguf"`
3. If `.safetensors` → read header (~1KB), inspect tensor key prefixes:
   - Keys contain `conditioner.`, `first_stage_model.`, `model.diffusion_model.` → `"full_checkpoint"`
   - Keys are transformer-only (`double_blocks`, `single_blocks`, `layers`, etc.) → `"transformer_only"`
4. Otherwise → treat as HF repo ID, `"hf_repo"`

### `assemble_pipeline(base_model_id, base_model_path, cls_name, emitter) -> Pipeline`

1. Look up assembly spec from `arch_config.gen_components`
2. Resolve each component's weights path from modl store via `_get_installed_path()`
3. Load each component individually:
   - **Transformer**: `TransformerClass.from_single_file(path, config=CONFIGS/..., torch_dtype=bf16)`
   - **Transformer (GGUF)**: Same but with `quantization_config=GGUFQuantizationConfig(...)`
   - **Text encoder**: `TEClass.from_pretrained(CONFIGS/..., torch_dtype=bf16)` then `load_state_dict()`
   - **Tokenizer**: `TokenizerClass.from_pretrained(CONFIGS/...)`  (config only, no weights)
   - **VAE**: `VAEClass.from_single_file(path, config=CONFIGS/..., torch_dtype=bf16)`
   - **Scheduler**: `SchedulerClass.from_pretrained(CONFIGS/...)`  (config only)
4. Construct: `PipelineClass(transformer=..., text_encoder=..., vae=..., ...)`
5. Move to CUDA: `pipe.to("cuda")`

Key principle: **respect the user's installed quantization**. If they have fp8, load fp8.
Don't cast to bf16.

### Updated `load_pipeline()` dispatch

```python
def load_pipeline(base_model_id, base_model_path, cls_name, emitter):
    model_source = base_model_path or resolve_model_path(base_model_id)
    fmt = detect_model_format(model_source)

    if fmt == "hf_directory":
        pipe = PipelineClass.from_pretrained(model_source, torch_dtype=torch.bfloat16)
    elif fmt == "full_checkpoint":
        pipe = PipelineClass.from_single_file(model_source, torch_dtype=torch.bfloat16)
    elif fmt in ("transformer_only", "gguf"):
        pipe = assemble_pipeline(base_model_id, base_model_path, cls_name, emitter)
    else:  # hf_repo
        pipe = PipelineClass.from_pretrained(model_source, torch_dtype=torch.bfloat16)

    return pipe.to("cuda")
```

---

## Changes to `arch_config.py`

### New `gen_components` format

Replace simple `{param: store_id}` with richer structure:

```python
"flux": {
    "pipeline_class": "FluxPipeline",
    "img2img_class": "FluxImg2ImgPipeline",
    "inpaint_class": "FluxInpaintPipeline",
    "gen_components": {
        "transformer": {
            "model_class": "FluxTransformer2DModel",
            "config_dir": "flux-dev-transformer",
        },
        "text_encoder": {
            "model_id": "clip-l",
            "model_class": "CLIPTextModel",
            "config_dir": "clip-l",
        },
        "tokenizer": {
            "model_class": "CLIPTokenizer",
            "config_dir": "clip-tokenizer",
        },
        "text_encoder_2": {
            "model_id": ["t5-xxl-fp8", "t5-xxl-fp16"],
            "model_class": "T5EncoderModel",
            "config_dir": "t5-xxl",
        },
        "tokenizer_2": {
            "model_class": "T5TokenizerFast",
            "config_dir": "t5-tokenizer",
        },
        "vae": {
            "model_id": "flux-vae",
            "model_class": "AutoencoderKL",
            "config_dir": "flux-vae",
        },
        "scheduler": {
            "model_class": "FlowMatchEulerDiscreteScheduler",
            "config_dir": "flux-dev-scheduler",
        },
    },
    ...
}
```

Fields:
- `model_id`: modl store ID to look up weights. `None` or absent = transformer (uses `base_model_path`). List = try each in order (fp8/fp16 fallback).
- `model_class`: diffusers/transformers class name to import and instantiate.
- `config_dir`: subdirectory under `python/modl_worker/configs/` with config.json, tokenizer files, etc.

### New entries needed

```python
# Add to ARCH_CONFIGS:
"qwen_image": {
    "pipeline_class": "QwenImagePipeline",
    "img2img_class": "QwenImageImg2ImgPipeline",
    "inpaint_class": "QwenImageInpaintPipeline",
    "gen_components": {
        "transformer": {"model_class": "QwenImageTransformer2DModel", "config_dir": "qwen-image-transformer"},
        "text_encoder": {"model_id": ["qwen-image-clip"], "model_class": "Qwen2_5_VLForConditionalGeneration", "config_dir": "qwen-image-text-encoder"},
        "tokenizer": {"model_class": "Qwen2Tokenizer", "config_dir": "qwen-image-tokenizer"},
        "vae": {"model_id": "qwen-image-vae", "model_class": "AutoencoderKLQwenImage", "config_dir": "qwen-image-vae"},
        "scheduler": {"model_class": "FlowMatchEulerDiscreteScheduler", "config_dir": "qwen-image-scheduler"},
    },
    ...
},

"flux2": {
    "pipeline_class": "Flux2Pipeline",
    "gen_components": {
        "transformer": {"model_class": "Flux2Transformer2DModel", "config_dir": "flux2-dev-transformer"},
        "text_encoder": {"model_id": "flux2-mistral-text-encoder", "model_class": "Mistral3ForConditionalGeneration", "config_dir": "flux2-mistral-te"},
        "tokenizer": {"model_class": "AutoProcessor", "config_dir": "flux2-mistral-tokenizer"},
        "vae": {"model_id": "flux2-vae", "model_class": "AutoencoderKLFlux2", "config_dir": "flux2-vae"},
        "scheduler": {"model_class": "FlowMatchEulerDiscreteScheduler", "config_dir": "flux2-dev-scheduler"},
    },
    ...
},

"flux2_klein": {
    "pipeline_class": "Flux2KleinPipeline",
    "gen_components": {
        "transformer": {"model_class": "Flux2Transformer2DModel", "config_dir": "flux2-klein-transformer"},
        "text_encoder": {"model_id": "flux2-qwen3-4b-text-encoder", "model_class": "Qwen3ForCausalLM", "config_dir": "flux2-qwen3-te"},
        "tokenizer": {"model_class": "Qwen2TokenizerFast", "config_dir": "flux2-qwen3-tokenizer"},
        "vae": {"model_id": "flux2-vae", "model_class": "AutoencoderKLFlux2", "config_dir": "flux2-vae"},
        "scheduler": {"model_class": "FlowMatchEulerDiscreteScheduler", "config_dir": "flux2-klein-scheduler"},
    },
    ...
},
```

### MODEL_REGISTRY additions

```python
MODEL_REGISTRY.update({
    "qwen-image-edit":   ("qwen_image",    "Qwen/Qwen-Image-Edit"),
    "flux2-dev":         ("flux2",         "black-forest-labs/FLUX.2-dev"),
    "flux2-klein-4b":    ("flux2_klein",   "black-forest-labs/FLUX.2-klein-4B"),
    "flux2-klein-9b":    ("flux2_klein",   "black-forest-labs/FLUX.2-klein-base-9B"),
})
```

### Bug fix: Chroma text encoder

Current (wrong): `gen_components: { text_encoder: "z-image-text-encoder" }`
Correct: Chroma uses T5 (same as Flux), not Qwen3.

---

## Bundled Config Files

Directory: `python/modl_worker/configs/`

### What goes in each config dir

Config dirs contain the small JSON/tokenizer files that diffusers needs to instantiate
a model from a bare safetensors. These are downloaded once from HF repos and committed.

**Estimated total size: ~25MB** (dominated by tokenizer vocabularies)

### Config dirs needed

Tier 1 (build now):
- `flux-dev-transformer/` — config.json (~1KB)
- `flux-schnell-transformer/` — config.json (~1KB, differs: `guidance_embeds: false`)
- `flux-dev-scheduler/` — scheduler_config.json
- `flux-schnell-scheduler/` — scheduler_config.json
- `flux-vae/` — config.json
- `clip-l/` — config.json
- `clip-tokenizer/` — vocab.json, merges.txt, tokenizer_config.json, special_tokens_map.json (~1.6MB)
- `t5-xxl/` — config.json
- `t5-tokenizer/` — tokenizer.json, tokenizer_config.json, special_tokens_map.json, spiece.model (~3.2MB)
- `zimage-transformer/` — config.json
- `zimage-scheduler/` — scheduler_config.json
- `zimage-vae/` — config.json
- `qwen3-text-encoder/` — config.json
- `qwen2-tokenizer/` — vocab.json, merges.txt, tokenizer.json, tokenizer_config.json (~16MB)

Tier 2 (build now, test later):
- `qwen-image-transformer/` — config.json
- `qwen-image-text-encoder/` — config.json + tokenizer files
- `qwen-image-tokenizer/` — tokenizer files
- `qwen-image-vae/` — config.json
- `qwen-image-scheduler/` — scheduler_config.json
- `flux2-dev-transformer/` — config.json
- `flux2-klein-transformer/` — config.json
- `flux2-vae/` — config.json
- `flux2-dev-scheduler/` — scheduler_config.json
- `flux2-klein-scheduler/` — scheduler_config.json
- `flux2-mistral-te/` — config.json + tokenizer files
- `flux2-mistral-tokenizer/` — tokenizer files
- `flux2-qwen3-te/` — config.json
- `flux2-qwen3-tokenizer/` — tokenizer files
- `chroma-transformer/` — config.json
- `chroma-scheduler/` — scheduler_config.json
- `chroma-tokenizer/` — spiece.model, tokenizer_config.json, etc.

### Fetch script

One-time script to download configs from HF repos:

```python
# scripts/fetch_component_configs.py
from huggingface_hub import hf_hub_download
import shutil

CONFIGS = {
    "clip-l": ("openai/clip-vit-large-patch14", ["config.json"]),
    "clip-tokenizer": ("openai/clip-vit-large-patch14", [
        "vocab.json", "merges.txt", "tokenizer_config.json", "special_tokens_map.json",
    ]),
    "t5-xxl": ("google/t5-v1_1-xxl", ["config.json"]),
    "t5-tokenizer": ("google/t5-v1_1-xxl", [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spiece.model",
    ]),
    # ... etc for each config dir
}

for config_dir, (repo, files) in CONFIGS.items():
    for fname in files:
        path = hf_hub_download(repo, fname)
        dest = f"python/modl_worker/configs/{config_dir}/{fname}"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy(path, dest)
```

---

## Implementation Steps

### Step 1: Fetch and commit config files (~25MB)

Write fetch script, run it, commit configs to repo.
Only Tier 1 configs are strictly needed for first test.

### Step 2: Update `arch_config.py`

- Restructure `gen_components` to richer format (model_class, config_dir, model_id)
- Fix Chroma text encoder bug (z-image-text-encoder → t5-xxl)
- Add new arch entries: `qwen_image` (fix pipeline_class from FluxPipeline to QwenImagePipeline),
  `flux2`, `flux2_klein`
- Add MODEL_REGISTRY entries for new models
- Add `resolve_gen_assembly()` function returning full component specs
- Keep existing `resolve_gen_components()` backward-compatible for training adapter

### Step 3: Add `detect_model_format()` to `gen_adapter.py`

Reads safetensors header (first ~1KB), inspects tensor key prefixes.
Returns: `"hf_directory"`, `"full_checkpoint"`, `"gguf"`, `"transformer_only"`, or `"hf_repo"`.

### Step 4: Add `assemble_pipeline()` to `gen_adapter.py`

The core new function. For each component in the assembly spec:
1. Resolve weights path from modl store
2. Load with appropriate class + bundled config
3. Construct pipeline from components
4. Return on CUDA

### Step 5: Update `load_pipeline()` dispatch

Replace current 3-way if/elif/else with format detection → strategy dispatch.

### Step 6: Test

**Priority order:**
1. `flux-dev` (fp8) — the user's immediate need
2. `z-image-turbo` — verify no regression (already works)
3. `flux-schnell` — if installed
4. SDXL — verify `from_single_file` still works (no regression)

**Later (when models installed):**
5. `qwen-image` — GGUF loading
6. `qwen-image-edit` — safetensors + edit pipeline
7. `flux2-dev` — new architecture
8. `flux2-klein-4b` — Klein pipeline
9. `chroma` — T5 text encoder fix

**Smoke test command:**
```bash
# Should assemble from local components (no HF download)
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 modl generate "a cat on mars" --base flux-dev
```

---

## Key Decisions

1. **Config files committed to repo** (not downloaded at runtime) — ensures offline operation,
   deterministic builds, and ~25MB is tiny compared to the models themselves.

2. **Format detection via safetensors header** — cheap (~1KB read), no downloads, no guessing.

3. **GGUF support via `GGUFQuantizationConfig`** — diffusers handles dequantization transparently.
   Extends reach to low-VRAM users (8GB+).

4. **Respect user's quantization** — fp8 transformer stays fp8, not upcast to bf16.
   Only VAE and non-quantized components get bf16.

5. **Backward compatible** — HF directory and full checkpoint paths unchanged.
   Assembly only kicks in for transformer-only safetensors (the modl store format).

6. **Training adapter unaffected** — `resolve_gen_components()` keeps returning simple
   `{param: path}` dict for the training code path.

---

## Risks / Open Questions

1. **Text encoder state_dict key mismatches** — bare safetensors from comfyanonymous repos
   may have different key names than what diffusers expects. Need to verify for each TE type
   and add key remapping if needed.

2. **Qwen2_5_VLForConditionalGeneration loading** — this is a full VL model, not just a
   text encoder. Loading from a single safetensors + config may require special handling.
   May need the full HF model directory with processor configs.

3. **Mistral3ForConditionalGeneration** — same concern as Qwen VL. These are large LLMs
   used as text encoders. Loading from bare safetensors may need investigation.

4. **AutoencoderKLQwenImage vs AutoencoderKL** — QwenImage uses a specialized VAE class.
   Need to verify `from_single_file` works with bundled config.

5. **Flux2 VAE (AutoencoderKLFlux2)** — different from Flux1 VAE. Cannot be shared.

6. **Tokenizer sizes** — Qwen2 tokenizer is ~16MB, might warrant git-lfs. But total configs
   at ~25MB are still manageable in regular git.

7. **ZImagePipeline Qwen3 TE** — the HF config may say `Qwen3ForCausalLM` but the pipeline
   expects `Qwen3Model` (base, not causal). Need to verify which class to use.
