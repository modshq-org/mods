# mods

**The opinionated toolkit for AI image generation.** Models, training, inference — one CLI.

`mods model pull flux-dev` downloads the model, its required VAE, its text encoders — everything — to the right folders, with verified hashes and compatibility checking. Then `mods train` fine-tunes a LoRA on your photos. Then `mods generate` creates images.

**[Website](https://mods.pedroalonso.net)** · **[Docs](https://mods.pedroalonso.net/docs)** · **[Model Registry](https://github.com/modshq-org/mods-registry)** · **[Issues](https://github.com/modshq-org/mods/issues)**

## Quick Start

```bash
# Install mods
curl -fsSL https://raw.githubusercontent.com/modshq-org/mods/main/install.sh | sh

# Or build from source
# git clone https://github.com/modshq-org/mods && cd mods && cargo install --path .

# First-time setup (auto-detects ComfyUI, A1111, etc.)
mods init

# Pull a model (auto-selects variant for your GPU)
mods model pull flux-dev

# See what's installed
mods model ls

# Search for LoRAs
mods model search "realistic" --type lora
```

## The Full Journey

```bash
# 1. Pull a base model
mods model pull flux-schnell

# 2. Prepare a training dataset
mods dataset create products --from ~/photos/my-products/

# 3. Train a LoRA
mods train --dataset products --base flux-schnell --name product-v1

# 4. Generate images (coming soon)
mods generate "a photo of OHWX on marble countertop" --lora product-v1
```

## How It Works

Mods keeps **one copy** of every model in a content-addressed store (`~/mods/store/`). Your tools see symlinks that point into the store.

```
~/mods/store/checkpoint/a1b2c3.../flux1-dev.safetensors   ← single file on disk
    ↑                       ↑
    │                       └── ~/A1111/models/Stable-diffusion/flux1-dev.safetensors (symlink)
    └── ~/ComfyUI/models/checkpoints/flux1-dev.safetensors (symlink)
```

Install once, use everywhere. No duplicate 24GB files across tools.

## Already Have Models?

If you already have models downloaded in ComfyUI or A1111, `mods model link` adopts the ones it recognizes — moves them into the store and replaces them with symlinks. Your tools keep working, nothing breaks.

```bash
# Adopt existing ComfyUI models
mods model link --comfyui ~/ComfyUI

# Or A1111
mods model link --a1111 ~/stable-diffusion-webui
```

What happens:
- Mods scans your model folders and hashes each file
- Files that match the registry are **moved** to `~/mods/store/` and replaced with symlinks
- Files mods doesn't recognize are **left untouched** (custom merges, community models, etc.)
- Your tools don't notice the difference — symlinks are transparent

After linking, `mods model pull` will automatically symlink new models into all your configured tools.

## Features

- **Dependency resolution** — `mods model pull flux-dev` installs required VAE, text encoders automatically
- **GPU-aware variant selection** — picks fp16/fp8/GGUF based on your VRAM
- **Content-addressed storage** — deduplicated, hash-verified downloads
- **Multi-tool support** — symlinks into ComfyUI, A1111, and more (InvokeAI planned)
- **Adopt existing models** — `mods model link` migrates your current library without re-downloading
- **Resumable downloads** — partial downloads resume automatically
- **Lock files** — `mods model export` / `mods model import` for reproducible environments
- **LoRA training** — opinionated presets (Quick/Standard/Advanced) powered by ai-toolkit
- **Managed runtime** — auto-installs Python, PyTorch, ai-toolkit — no conda/venv juggling
- **Dataset management** — organize, validate, and caption training images

## Commands

### System

| Command | Description |
|---------|-------------|
| `mods init` | First-time setup — detect tools, configure storage |
| `mods doctor` | Check for broken symlinks, missing deps, corrupt files |
| `mods config [key] [value]` | View or update configuration |
| `mods auth <provider>` | Configure authentication (HuggingFace, Civitai) |
| `mods upgrade` | Update mods CLI to the latest release |

### Models (`mods model`)

| Command | Description |
|---------|-------------|
| `mods model pull <id>` | Download a model with all dependencies |
| `mods model rm <id>` | Remove an installed model |
| `mods model ls` | List installed models |
| `mods model info <id>` | Show detailed info about a model |
| `mods model search <query>` | Search the registry |
| `mods model popular` | Show trending models |
| `mods model link` | Adopt existing tool model folders |
| `mods model update` | Fetch latest registry index |
| `mods model space` | Show disk usage breakdown |
| `mods model gc` | Remove unreferenced files from the store |
| `mods model export` / `import` | Shareable lock files for reproducible setups |

### Training (`mods train`)

| Command | Description |
|---------|-------------|
| `mods train` | Train a LoRA (interactive or with flags) |
| `mods train setup` | Install training dependencies (ai-toolkit + PyTorch) |

### Datasets (`mods dataset`)

| Command | Description |
|---------|-------------|
| `mods dataset create <name> --from <dir>` | Create a managed dataset from images |
| `mods dataset ls` | List all managed datasets |
| `mods dataset validate <name>` | Validate a dataset for training |

### Runtime (`mods runtime`)

| Command | Description |
|---------|-------------|
| `mods runtime install` | Install managed Python runtime |
| `mods runtime status` | Show runtime installation status |
| `mods runtime doctor` | Run runtime health checks |
| `mods runtime bootstrap` | Bootstrap environment and install deps |
| `mods runtime upgrade` | Upgrade runtime to latest version |
| `mods runtime reset` | Reset runtime state |

## Variant Selection

Models come in multiple variants. Mods picks the best one for your GPU automatically:

| VRAM | Variant | Notes |
|------|---------|-------|
| 24GB+ | fp16 | Full quality |
| 12-23GB | fp8 | Slight quality reduction |
| 8-11GB | gguf-q4 | Quantized, needs GGUF loader |
| <8GB | gguf-q2 | Lower quality, functional |

Override with `mods model pull flux-dev --variant fp8`.

## License

MIT
