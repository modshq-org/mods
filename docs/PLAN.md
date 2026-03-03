# modl — Plan & Status

> Last updated: 2026-03-03
> Single source of truth. Everything else in `docs/` is reference material.

## What modl is

The sane, opinionated way to run diffusion locally.

1. **Opinionated model installer** — `modl pull flux-dev`, deps handled, no thinking
2. **Opinionated LoRA trainer** — presets (quick/standard/advanced), sensible defaults
3. **Opinionated generator** — simple flags, seeds, batch, size presets, LoRA stacking

CLI-first. Rust binary + managed Python runtime. Local-first.

---

## Current Status

**~15K Rust LOC, ~2K Python LOC. 55 unit tests passing. On `main` branch.**

| Area | Status | Notes |
|------|--------|-------|
| Model pull/ls/rm/search/info | ✅ Solid | Registry (68 models), HF direct pulls (`hf:` prefix), content-addressed store |
| Model link (ComfyUI/A1111) | ✅ Solid | Auto-detect layouts, cross-device fallback |
| HuggingFace integration | ✅ Done | `modl pull hf:owner/repo`, HF fallback in search |
| Config / Auth / GPU detect | ✅ Done | YAML config, HF/CivitAI tokens, NVML + nvidia-smi |
| Dataset create/validate/caption | ✅ Done | Florence-2/BLIP auto-captioning, tag, resize |
| Training (presets + executor) | ✅ Working | SDXL LoRA trained successfully, previews generated |
| Generation (CLI) | ✅ Built | Flux/SDXL/SD1.5 via diffusers, LoRA loading |
| Output management | ✅ Done | `modl outputs` list/show/open/search |
| Runtime bootstrap | ✅ Done | Python venv + ai-toolkit install |
| Doctor/GC/Export/Import | ✅ Done | Orphan detection, repair, lockfile round-trip |
| `modl upgrade` | ✅ Done | Self-update from GitHub releases |

---

## Architecture

```
modl CLI (Rust, single binary)
    │
    ├── presets::resolve_params()   ── pure logic, no I/O
    ├── dataset::validate()         ── filesystem scan
    ├── gpu::detect()               ── NVML + nvidia-smi
    │
    ▼
TrainJobSpec / GenerateJobSpec      ◄── serialization boundary
    │
    ▼
LocalExecutor                       ◄── spawn Python, parse JSONL events
    │
    ▼
Python runtime (modl_worker/)
    ├── train_adapter.py            ── spec → ai-toolkit YAML
    ├── gen_adapter.py              ── diffusers pipeline loading
    ├── caption_adapter.py          ── Florence-2/BLIP
    └── protocol.py                 ── JSON-line events over stdout
```

---

## Phases

### Phase 1 — Bulletproof install + setup ✅

*Done.* One `curl | sh` install. `modl doctor` catches broken symlinks, missing
deps, corrupt files. `modl pull` resolves dependencies. GPU auto-detection picks
the right variant.

Remaining polish:
- [ ] CUDA compatibility edge cases (test on more machines)
- [ ] `modl init` wizard for first-time setup

### Phase 2 — Validate full train→generate flow 🔄

The pipeline exists end-to-end. SDXL LoRA training works with preview generation.

- [x] SDXL LoRA training (confirmed working)
- [x] Training previews / samples
- [x] Dataset annotation
- [ ] Flux training E2E validation
- [ ] Generation E2E validation (`modl generate` → image on disk)
- [ ] Fix integration issues that surface on GPU

### Phase 3 — Multi-arch training support

Curated ai-toolkit configs for models people actually train on.
See [multi-arch-training-plan.md](multi-arch-training-plan.md) for full gap analysis.

Priority order (by real-world demand):

| # | Model | Status | Effort |
|---|-------|--------|--------|
| 1 | flux-dev / flux-schnell | ✅ Config ready | — |
| 2 | sdxl / sd1.5 | ✅ Working | — |
| 3 | z-image-turbo | 🟡 Config ready | E2E test |
| 4 | chroma | 🟡 Config ready | E2E test |
| 5 | Flux Kontext | ❌ Needs paired dataset support | ~1h |
| 6 | FLUX.2 | ❌ Needs arch entry | ~30min |
| 7 | Qwen-Image | ❌ Needs qtype + quant for 24GB | ~1h |

### Phase 4 — Polish & batch

- [ ] Batch generation (`modl generate --batch prompts.txt`)
- [ ] Reproducible export (`modl outputs export <id>` → full YAML spec)
- [ ] Registry curation: core tier (flux-dev, flux-schnell, sdxl) vs experimental
- [ ] VRAM-aware config tuning (auto quantize/offload based on GPU)

### Phase 5 — Persistent worker (performance)

Eliminate 20-45s cold start on repeated `modl generate` calls.
Python daemon with LRU model cache, LoRA hot-swap.
See [specs/persistent-worker.md](specs/persistent-worker.md) for full spec.

- [ ] Python serve mode (`modl_worker serve` on Unix socket)
- [ ] Rust worker management (auto-spawn, health check, idle timeout)
- [ ] Model cache with LRU eviction

### Phase 6 — Web UI (`modl serve`)

Prompt-first generate page. Training dashboard. Gallery.
Build only after the CLI flow is rock-solid.
See [archive/ui-architecture.md](archive/ui-architecture.md) for product spec.

- [ ] REST + WebSocket API
- [ ] Static UI compiled into binary
- [ ] Generate page (prompt → image, LoRA selector)
- [ ] Output gallery

### Phase 7 — Cloud training (`--cloud`)

`modl train --cloud` submits to a managed API. Cloud inference deferred.
See [archive/cloud-plan.md](archive/cloud-plan.md) for architecture and pricing model.

- [ ] modl API service (auth, billing, job dispatch)
- [ ] Modal GPU backend
- [ ] CloudExecutor.submit() implementation

### Backlog — Not planned

| Feature | Why not now |
|---------|------------|
| Video model training (Wan, LTX) | Different pipeline, defer until image is solid |
| CivitAI direct pulls | Need API key setup, lower priority than HF |
| Cloud inference | Cold start + keep_warm economics are brutal |
| DAM / tagging / collections | Filesystem + `modl outputs search` is enough |
| Node/graph editor | ComfyUI owns this, don't compete |
| Multi-provider cloud (RunPod, etc.) | Get one provider working first |

---

## Reference docs

| Doc | What it covers | Status |
|-----|---------------|--------|
| [multi-arch-training-plan.md](multi-arch-training-plan.md) | ai-toolkit arch configs, per-model gaps | Active — Phase 3 guide |
| [specs/aitoolkit-mapping.md](specs/aitoolkit-mapping.md) | TrainJobSpec → ai-toolkit YAML field mapping | Implemented, canonical |
| [specs/jobs-schema-v1.md](specs/jobs-schema-v1.md) | Job/event/artifact JSON schemas | Implemented, canonical |
| [specs/worker-protocol.md](specs/worker-protocol.md) | JSONL protocol between Rust and Python | Implemented, canonical |
| [specs/execution-target.md](specs/execution-target.md) | Executor trait contract | Implemented (local), stubbed (cloud) |
| [specs/persistent-worker.md](specs/persistent-worker.md) | Daemon architecture for fast generation | Phase 5 spec |
| [archive/ui-architecture.md](archive/ui-architecture.md) | Web UI product spec | Phase 6 reference |
| [archive/cloud-plan.md](archive/cloud-plan.md) | Cloud platform architecture + pricing | Phase 7 reference |
| [archive/runtime-architecture.md](archive/runtime-architecture.md) | ComfyUI sidecar / YAML workflow vision | Aspirational, not planned |
| [archive/capability-model.md](archive/capability-model.md) | Cloud auth/quota gating | Phase 7 detail |
| [archive/runtime-profiles.md](archive/runtime-profiles.md) | Reproducible runtime manifests | Over-engineered, current venv works |

---

## What modl is NOT

- Not a node editor (use ComfyUI)
- Not a marketplace (use CivitAI)
- Not a hosted service (you run it)
- Not infinitely configurable (three presets, Advanced gives full YAML)
- Not a DAM (filesystem + metadata JSON is enough)
- Not a cloud platform (maybe later, not now)
