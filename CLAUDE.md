# CLAUDE.md — Modl

## What is Modl?

Modl is a local-first image generation toolkit. One binary. Pull models from a registry, generate images, train LoRAs, organize outputs. Like Ollama for image generation.

```
curl -fsSL https://modl.run/install.sh | bash
modl pull flux-schnell
modl generate "a cat on mars"
```

The core loop: **pull models -> generate images -> train LoRAs -> manage outputs**. Everything through CLI, backed by a single Rust binary + Python worker for GPU inference. Web UI is in development on the `feat/ui` branch.

**GitHub org:** [github.com/modl-org](https://github.com/modl-org)

## Feature Status

| Feature | Status |
|---------|--------|
| txt2img | Built (CLI) |
| img2img + inpainting | Built (CLI: `--init-image`, `--mask`) |
| LoRA application | Built (CLI: `--lora`, `--lora-strength`) |
| LoRA training | Built (CLI: `modl train`) |
| Model registry + pull | Built |
| Outputs gallery | Built (CLI: `modl outputs`) |
| Image analysis | Built (score, detect, compare, segment, face-restore, upscale, remove-bg) |
| Prompt enhancement | Built (`modl enhance`) |
| Web UI | In progress (`feat/ui` branch) |
| Video gen | Not built |

Supported architectures: Flux, Flux Schnell, SDXL, SD1.5, Z-Image, Z-Image Turbo, Chroma, Qwen-Image.

## Tech Stack

- **Language:** Rust (stable toolchain)
- **CLI:** `clap` v4 with derive macros
- **Terminal UX:** `indicatif`, `console`, `dialoguer`, `comfy-table`
- **HTTP:** `reqwest` with `stream` + `rustls-tls`, `tokio` async runtime
- **Web server:** `axum` 0.8 (stub on main, full server on `feat/ui`)
- **Database:** `rusqlite` with `bundled` feature
- **GPU detection:** `nvml-wrapper` with `nvidia-smi` fallback
- **Dirs:** `dirs` crate for platform-specific paths

### Two-Process Architecture

| Process | Language | Role |
|---------|----------|------|
| **modl** (Rust) | Rust | CLI, job orchestration, DB, model management |
| **Worker** | Python | Inference, training, analysis, VRAM management |

Communication: subprocess stdout (JSON events) or Unix socket (persistent worker).

## Source Code Organization

```
src/
├── main.rs              Entry point, tokio runtime
├── cli/                  One file per command
│   ├── mod.rs            Clap definitions + dispatch
│   ├── install.rs        modl pull
│   ├── generate.rs       modl generate
│   ├── train.rs          modl train
│   ├── datasets.rs       modl dataset *
│   ├── outputs.rs        modl outputs *
│   ├── analysis.rs       Shared analysis worker spawner
│   ├── score.rs          modl score
│   ├── detect.rs         modl detect
│   ├── compare.rs        modl compare
│   ├── segment.rs        modl segment
│   ├── face_restore.rs   modl face-restore
│   ├── upscale.rs        modl upscale
│   ├── remove_bg.rs      modl remove-bg
│   ├── enhance.rs        modl enhance
│   ├── serve.rs          modl serve
│   ├── worker.rs         modl worker *
│   └── ...
├── core/                 Business logic. No terminal output.
│   ├── config.rs         ~/.modl/config.yaml
│   ├── db.rs             SQLite (models, jobs, artifacts, favorites, studio sessions)
│   ├── store.rs          Content-addressed storage + hash verification
│   ├── manifest.rs       Registry manifest types
│   ├── registry.rs       Load/search index.json
│   ├── resolver.rs       Dependency resolution
│   ├── download.rs       Resilient HTTP downloads
│   ├── executor.rs       Executor trait + LocalExecutor
│   ├── job.rs            Serializable job specs (Generate, Train, Score, etc.)
│   ├── presets.rs        Training parameter resolution
│   ├── runtime.rs        Managed Python runtime
│   ├── dataset.rs        Dataset management
│   ├── outputs.rs        Service layer for generated images
│   ├── install.rs        Service layer for model installation
│   ├── training.rs       Training worker path resolution
│   ├── gpu.rs            GPU detection
│   ├── symlink.rs        Symlink management
│   ├── enhance.rs        Prompt enhancement
│   ├── llm.rs            LlmBackend trait (builtin/cloud/local)
│   ├── agent.rs          Tool-use agent loop (experimental)
│   ├── agent_tools.rs    Agent tool implementations
│   └── update_check.rs   Background CLI update check
├── auth/                 HuggingFace + Civitai token management
├── compat/               ComfyUI, A1111, InvokeAI path layouts
└── ui/                   Web UI (stub on main, full on feat/ui)
    ├── server.rs         Axum server + routes
    ├── routes/           API route handlers
    └── dist/             Built frontend (stub on main)
```

## CLI Commands

All commands are top-level — there is NO `modl model` subcommand. Run `modl cli-schema` for machine-readable JSON of all commands.

### Core workflow
- `modl pull <id>` — Download model with all dependencies
- `modl generate "prompt"` — Generate images (`--base`, `--lora`, `--lora-strength`, `--size`, `--steps`, `--guidance`, `--seed`, `--init-image`, `--mask`, `--strength`)
- `modl train` — Train a LoRA (`--base`, `--dataset`, `--name`, `--lora-type` [required], `--preset`, `--steps`)
- `modl dataset create|ls|rm|caption|tag|resize|prepare|validate` — Dataset management

### Model management
- `modl ls`, `modl rm`, `modl info`, `modl search`, `modl update`, `modl link`, `modl gc`

### Image analysis
- `modl score`, `modl detect`, `modl compare`, `modl segment`, `modl face-restore`, `modl upscale`, `modl remove-bg`

### System
- `modl init`, `modl doctor`, `modl config`, `modl auth`, `modl upgrade`, `modl serve`
- `modl worker start|stop|status` — Persistent GPU worker
- `modl enhance "prompt"` — AI prompt expansion
- `modl llm pull|chat|ls` — LLM management (experimental)
- `modl outputs ls|show|open|search|fav|unfav|rm` — Output management

## Key Concepts

### Content-Addressed Storage
Models stored by SHA256 in `~/modl/store/<type>/<hash>/<filename>`. Symlinks point from tool folders (ComfyUI, A1111) into the store.

### Dependency Resolution
`modl pull flux-dev` installs checkpoint + VAE + text encoders automatically via manifest `requires` declarations.

### Variant Selection
Auto-selects fp16/fp8/GGUF based on GPU VRAM. Override: `modl pull flux-dev --variant fp8`.

### Executor Trait
`LocalExecutor` spawns Python worker subprocess. `CloudExecutor` is a stub. Both CLI and UI use the same executor.

### Specification-Driven Execution
`TrainJobSpec`, `GenerateJobSpec`, etc. are serializable YAML. Written to disk before execution, stored in DB for provenance.

## Design Principles

### CLI is the master for all operations
Every operation must be available as a CLI command first. The web UI calls the same core logic.

When adding a new feature:
1. Implement in `src/core/` (business logic)
2. Expose as CLI command in `src/cli/`
3. (Later) Wire into web UI

### Service layer abstraction
CLI and UI handlers must NOT talk to DB/filesystem directly. All mutations go through `src/core/` services.

### CLI handlers should be thin
Parse args, call `core/` service, format output. Business logic belongs in `src/core/`.

## Code Style

- `clap` derive macros for CLI (not builder pattern)
- `anyhow::Result` in CLI layer, `thiserror` in core
- Async only where needed (downloads, HTTP). File I/O synchronous.
- `reqwest` streaming for large downloads
- SQLite for state — NOT JSON files
- `serde` derive on all manifest/config structs

## Implementation Notes

### Worker communication
- One-shot: subprocess stdout (JSON `JobEvent` lines)
- Persistent: Unix socket (`python/modl_worker/serve.py`)
- Analysis commands route through persistent worker for model caching

### Download resilience
Resume (HTTP Range), retry (3x exponential backoff), SHA256 verification, partial cleanup on failure.

### Port killing (SSH safety)
**Always** use `lsof -sTCP:LISTEN` — never plain `lsof -ti :PORT` (kills SSH port-forwarding PIDs).

### Training via SSH
SIGHUP kills child process. Use tmux/screen for long training runs.

## Python Worker

```
python/modl_worker/
├── main.py              CLI entrypoint (argparse subcommands)
├── serve.py             Persistent daemon (Unix socket, model caching)
├── protocol.py          EventEmitter (JSON events to stdout/socket)
└── adapters/
    ├── gen_adapter.py       Image generation (diffusers pipelines)
    ├── train_adapter.py     LoRA training (ai-toolkit)
    ├── arch_config.py       Architecture configs + MODEL_REGISTRY
    ├── score_adapter.py     Aesthetic scoring
    ├── detect_adapter.py    Face detection
    ├── compare_adapter.py   CLIP similarity
    ├── segment_adapter.py   SAM segmentation
    ├── face_restore_adapter.py  CodeFormer
    ├── upscale_adapter.py   Real-ESRGAN / Spandrel
    └── remove_bg_adapter.py BiRefNet background removal
```

9 model architectures: flux, flux_schnell, sdxl, sd15, zimage, zimage_turbo, chroma, qwen_image (defined in `arch_config.py`).

## What is NOT built yet

- **Web UI** — In progress on `feat/ui` branch (React 19 + TypeScript + Tailwind)
- **CloudExecutor** — Stub only
- **Local LLM inference** — llama-cpp-2 integration pending
- **Video generation** — Not started
