# modl

**The easiest way to generate images and train LoRAs on your own GPU.**

One command to install, one command to generate, one command to train. No Python environments, no dependency hell, no 2-hour YouTube tutorials.

```bash
curl -fsSL https://modl.run/install.sh | sh
modl pull flux-schnell
modl generate "a cat on mars"
```

**[Website](https://modl.run)** · **[Docs](https://modl.run/docs)** · **[Model Registry](https://github.com/modl-org/modl-registry)** · **[Issues](https://github.com/modl-org/modl/issues)**

---

## Why modl?

**No glue code.** One binary handles model downloads, dependency resolution, image generation, LoRA training, and output management. No separate tools to install, no configs to write.

**Smart model management.** Models are stored once in a content-addressed store. ComfyUI, A1111, and other tools see symlinks — no duplicate 24GB files.

**GPU-aware.** Automatically picks the right model variant (fp16, fp8, quantized) for your VRAM. A 4090 gets full quality. An 8GB card still works.

**Train LoRAs in one command.** Point it at a folder of images, pick a base model, and go. Auto-captioning, dataset prep, and sensible defaults included.

---

## Quick Start

```bash
# Install
curl -fsSL https://modl.run/install.sh | sh

# Pull a model (auto-selects variant for your GPU)
modl pull flux-dev

# Generate
modl generate "a photo of a mountain lake at sunset" --base flux-dev
```

Or do everything at once:

```bash
curl -fsSL https://modl.run/install.sh | sh -s -- --quick
```

This installs modl, pulls a starter model, and launches the web UI.

---

## Train a LoRA

```bash
# Prepare dataset (auto-captions your images)
modl dataset prepare my-product --from ~/photos/product-shots/

# Train
modl train --dataset my-product --base flux-dev --name product-v1 --lora-type object

# Generate with your LoRA
modl generate "a photo of OHWX on marble countertop" --lora product-v1
```

---

## Web UI

```bash
modl serve
```

Full generation, training, and output management in the browser at `http://localhost:3333`. Same engine as the CLI.

Install as a system service (starts on boot):

```bash
modl serve --install-service
```

---

## Supported Models

| Family | Model | Params | VRAM (fp8) | Generate | Edit | Train |
|--------|-------|--------|------------|:--------:|:----:|:-----:|
| **Flux 1** | Flux Dev | 12B | 16 GB | yes | | yes |
| | Flux Schnell | 12B | 16 GB | yes | | yes |
| **Flux 2** | Flux 2 Dev | 24B | 24 GB | yes | yes | |
| | Klein 4B | 4B | 10 GB | yes | yes | yes |
| | Klein 9B | 9B | 16 GB | yes | yes | yes |
| **Z-Image** | Z-Image | 6B | 12 GB | yes | | yes |
| | Z-Image Turbo | 6B | 12 GB | yes | | yes |
| **Qwen Image** | Qwen Image | 20B | 20 GB | yes | | yes |
| | Qwen Image Edit | 20B | 20 GB | | yes | |
| **Legacy** | SDXL | 3.5B | 5 GB | yes | | yes |

**Generate** = text-to-image. **Edit** = instruction-based image editing (`modl edit "make it blue" --image photo.jpg`). **Train** = LoRA fine-tuning.

```bash
modl pull flux-schnell     # fast, 4-step generation
modl pull flux-dev         # high quality, best for training
modl pull z-image-turbo    # lightweight, fast
```

---

## Already Have Models?

```bash
modl link --comfyui ~/ComfyUI
modl link --a1111 ~/stable-diffusion-webui
```

Modl scans your model folders, hashes files, and moves recognized models into the store — replacing them with symlinks. Your tools keep working, nothing breaks.

---

## Image Tools

```bash
modl edit "make the sky purple" --image photo.jpg    # AI image editing
modl upscale photo.jpg                                # 4x upscale
modl remove-bg photo.jpg                              # transparent PNG
modl face-restore photo.jpg                           # fix faces
modl score photo.jpg                                  # aesthetic quality
```

---

## Docker

```bash
docker run --gpus all -p 3333:3333 -v modl-data:/workspace ghcr.io/modl-org/modl:latest
```

Set `MODEL=flux-schnell` to auto-pull a model on first boot. Models persist on the volume across restarts.

---

## Architecture

Single Rust binary for speed and distribution. Managed Python runtime for GPU compute. No external dependencies to install.

Full CLI reference: **[modl.run/docs](https://modl.run/docs)**

---

## License

MIT
