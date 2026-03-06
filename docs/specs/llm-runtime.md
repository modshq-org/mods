# LLM / VL Runtime Architecture

> **STATUS: PLANNED — Phase 9.** Decisions captured from architecture review.

---

## Runtime decision: llama.cpp

**llama.cpp** (via `llama-cpp-python` bindings), not transformers/vllm.

Same embedding pattern as:
- **ollama** — bundles llama.cpp, manages GGUF models, exposes API
- **ai-toolkit embed** — managed Python runtime, spawned from Rust CLI

### Why llama.cpp

| Concern | llama.cpp | transformers/vllm |
|---------|-----------|-------------------|
| Quantization | Native GGUF (Q4-Q8), first-class | Bolt-on (bitsandbytes, GPTQ) |
| 9B-30B sweet spot | Exactly its target | Overkill for small models |
| GPU memory | Very efficient | Heavier overhead |
| VL support | Qwen2.5-VL, LLaVA via mmproj | Broader but heavier |
| Startup time | Fast | Slow (torch import ~3s) |
| Dependency weight | Single .so + Python bindings | PyTorch + transformers + tokenizers |
| Model format | GGUF files (fit modl store) | HF repos (dirs with configs) |

### Why not ollama

Ollama wraps llama.cpp but adds: a separate daemon process, its own model
management (conflicts with modl's store), Go binary dependency, API layer we
don't need. We already have the model management — just embed the runtime
directly, same as we embed ai-toolkit for diffusion.

---

## Architecture

```
modl CLI (Rust)
    │
    ├── modl enhance "prompt"          (text LLM)
    ├── modl dataset caption --model qwen-vl   (vision-language)
    │
    ▼
LlmExecutor (same Executor trait pattern)
    │
    ├── LocalLlmExecutor
    │   ├── Persistent worker mode (preferred)
    │   │   └── llama-cpp-python server on Unix socket
    │   │       ├── Model cache (same LRU pattern as diffusion worker)
    │   │       ├── GGUF model loaded from modl store
    │   │       └── mmproj loaded for VL models
    │   │
    │   └── One-shot mode (fallback)
    │       └── spawn python -m modl_worker.main llm --config <spec>
    │
    ├── ApiLlmExecutor
    │   ├── OpenAI-compatible API (Ollama, vLLM, LM Studio)
    │   └── Configured via ~/.modl/config.yaml
    │
    └── CloudLlmExecutor
        └── modl API → cloud LLM endpoint
```

### Worker integration

The LLM worker follows the same pattern as the diffusion persistent worker:

- **Socket**: `~/.modl/llm-worker.sock` (separate from `worker.sock`)
- **Protocol**: Same JSONL event stream (protocol.py)
- **Cache**: Keep model loaded in VRAM between requests
- **Idle timeout**: Auto-shutdown after 5min idle (LLM models are fast to reload)
- **GPU coexistence**: Small LLMs (4B Q4 ~3GB) can coexist with diffusion
  pipeline on 24GB+ GPUs. Larger LLMs (30B) need exclusive access.

### Model registry

LLM/VL models go in `modl-registry/manifests/` as new asset types:

```yaml
# manifests/language_models/qwen3.5-4b-instruct.yaml
id: qwen3.5-4b-instruct
name: "Qwen 3.5 4B Instruct"
type: language_model
architecture: qwen3

variants:
  - id: q4-k-m
    file: qwen3.5-4b-instruct-Q4_K_M.gguf
    url: https://huggingface.co/...
    sha256: "..."
    size: 2800000000
    format: gguf
    precision: q4_k_m
    vram_required: 3072
    vram_recommended: 4096

  - id: q8-0
    file: qwen3.5-4b-instruct-Q8_0.gguf
    url: https://huggingface.co/...
    sha256: "..."
    size: 4600000000
    format: gguf
    precision: q8_0
    vram_required: 5120
    vram_recommended: 6144

defaults:
  context_length: 8192
  temperature: 0.7

tags: [llm, instruct, multilingual]
```

For VL models, add mmproj as a dependency:

```yaml
# manifests/vision_language/qwen2.5-vl-7b.yaml
id: qwen2.5-vl-7b
name: "Qwen 2.5 VL 7B"
type: vision_language
architecture: qwen2_vl

variants:
  - id: q4-k-m
    file: qwen2.5-vl-7b-Q4_K_M.gguf
    ...

requires:
  - id: qwen2.5-vl-7b-mmproj
    type: clip_vision
    reason: "Vision projection model for image understanding"
```

### VRAM budget (coexistence targets)

| Config | Diffusion | LLM | Total | GPU |
|--------|-----------|-----|-------|-----|
| Flux fp8 + Qwen 4B Q4 | ~12GB | ~3GB | ~15GB | 24GB OK |
| Z-Image bf16 + Qwen 4B Q4 | ~14GB | ~3GB | ~17GB | 24GB OK |
| Flux fp8 + Qwen 9B Q4 | ~12GB | ~6GB | ~18GB | 24GB tight |
| Z-Image bf16 + Qwen 30B Q4 | ~14GB | ~18GB | ~32GB | needs 48GB or exclusive |

### Use cases (priority order)

1. **Prompt enhance** — short phrase to rich prompt. Single LLM call.
2. **Dataset captioning** — VL model describes training images.
   Replace Florence-2 with Qwen2.5-VL for instruction-following captions.
3. **Story to batch** — "10 fantasy scenes" → 10 prompts → batch generate.
4. **Caption critique** — review and improve training captions.

### Python adapter

```
modl_worker/
├── adapters/
│   ├── llm_adapter.py      # llama-cpp-python inference
│   └── ...
├── llm_serve.py             # Persistent LLM worker (Unix socket)
└── ...
```

`llm_adapter.py` wraps `llama-cpp-python`:
- `run_completion(spec, emitter)` — text completion
- `run_chat(spec, emitter)` — chat completion with system prompt
- `run_vision(spec, emitter)` — image + prompt → text (VL models)

All emit standard JSONL events via the existing protocol.

### Config

```yaml
# ~/.modl/config.yaml
llm:
  # Default model for enhance/caption (auto-selected by VRAM if not set)
  model: qwen3.5-4b-instruct
  variant: q4-k-m

  # Optional: use external API instead of local model
  # api:
  #   url: http://localhost:11434/v1   # Ollama
  #   model: qwen3.5:4b
```

### Graceful degradation

```
Local GPU LLM (best quality, ~3GB VRAM)
    ↓ no VRAM / model not pulled
Local CPU LLM (slow but works, via llama.cpp CPU mode)
    ↓ too slow / not installed
Builtin rules (existing PromptEnhancer, zero deps)
    ↓ user has cloud auth
Cloud API (modl cloud or user's own API)
```

The enhance button always works. Quality scales with available resources.

---

## Implementation plan

1. Add `language_model` and `vision_language` asset types to registry schema
2. Add GGUF model manifests to modl-registry
3. Add `llama-cpp-python` to the managed Python runtime
4. Implement `llm_adapter.py` (completion + chat + vision)
5. Implement `llm_serve.py` (persistent worker on separate socket)
6. Wire up `modl enhance` to use local LLM backend
7. Wire up `modl dataset caption --model qwen-vl` to use VL backend
8. Add VRAM coexistence checks to GPU resource manager

## Open questions

- **llama.cpp build**: bundle pre-built wheels (llama-cpp-python has CUDA wheels
  on PyPI) or compile from source in `modl runtime install`? Pre-built is
  simpler but may lag behind llama.cpp releases.
- **Context length**: 8K is fine for enhance/caption. Story mode may need 16K+.
  GGUF models support dynamic context via rope scaling.
- **Streaming**: enhance could stream tokens to the UI for responsiveness.
  The SSE infrastructure already exists.
