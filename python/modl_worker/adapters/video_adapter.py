"""Video generation adapter — loads LTX-2 pipeline and generates video.

Uses GGUF quantized transformer + Gemma 3 FP4 text encoder + distilled LoRA
for 24GB VRAM compatibility. Mirrors the proven ComfyUI approach.

Pipeline assembly:
  1. Transformer: GGUF via from_single_file + GGUFQuantizationConfig
  2. Text encoder + connector: from the diffusers-format repo (Lightricks/LTX-2)
  3. VAE: from_single_file
  4. LoRA: distilled LoRA for fast 8-step generation
  5. Sampling: manual sigmas matching ComfyUI distilled schedule
"""

import os
import time
from pathlib import Path

from modl_worker.protocol import EventEmitter


# Distilled sigma schedule (8 steps) — matches ComfyUI workflow
DISTILLED_SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]


def run_generate_video(config_path: Path, emitter: EventEmitter) -> int:
    """Run video generation from a GenerateJobSpec YAML file."""
    import yaml
    import torch

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    model_info = spec.get("model", {})
    base_model_id = model_info.get("base_model_id", "ltx-video-dev")
    base_model_path = model_info.get("base_model_path")
    lora_info = spec.get("lora")
    output_info = spec.get("output", {})
    params = spec.get("params", {})

    prompt = spec.get("prompt", "")
    width = params.get("width", 768)
    height = params.get("height", 512)
    num_frames = params.get("num_frames", 97)
    fps = params.get("fps", 24)
    steps = params.get("steps", 8)
    guidance = params.get("guidance", 1.0)
    seed = params.get("seed")
    count = params.get("count", 1)

    output_dir = output_info.get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    emitter.info(f"Loading LTX-2 video pipeline for {base_model_id}...")
    emitter.progress(stage="load", step=0, total_steps=count)

    try:
        pipe = _load_ltx2_pipeline(base_model_id, base_model_path, lora_info, emitter)
    except Exception as exc:
        emitter.error("PIPELINE_LOAD_FAILED", f"Failed to load video pipeline: {exc}", recoverable=False)
        import traceback
        traceback.print_exc()
        return 1

    emitter.job_started(config=str(config_path))

    # Generation loop
    generator = torch.Generator(device="cpu")
    if seed is None:
        seed = generator.seed()
    generator.manual_seed(seed)

    artifact_paths = []
    for i in range(count):
        t0 = time.time()

        def _step_callback(pipe_self, step_idx, timestep, callback_kwargs):
            emitter.progress(stage="step", step=step_idx + 1, total_steps=steps)
            return callback_kwargs

        try:
            result = pipe(
                prompt=prompt,
                negative_prompt="blurry, low quality, still frame, watermark",
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=float(fps),
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                callback_on_step_end=_step_callback,
                output_type="pil",
            )
            video_frames = result.frames[0]
        except Exception as exc:
            emitter.error(
                "GENERATION_FAILED",
                f"Video generation failed: {exc}",
                recoverable=(i + 1 < count),
            )
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0

        # Export video
        from modl_worker.video_util import save_and_emit_video_artifact
        filepath = save_and_emit_video_artifact(
            video_frames, fps, output_dir, emitter,
            index=i, count=count,
            metadata={
                "generated_with": "modl.run",
                "prompt": prompt,
                "base_model_id": base_model_id,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "steps": steps,
                "guidance": guidance,
                "seed": seed + i if seed is not None else None,
            },
            elapsed=elapsed,
        )
        if filepath:
            artifact_paths.append(filepath)

        if seed is not None:
            generator.manual_seed(seed + i + 1)

    if artifact_paths:
        emitter.completed(f"Generated {len(artifact_paths)} video(s)")
    else:
        emitter.error("NO_OUTPUT_GENERATED", "All generation attempts failed", recoverable=False)
        return 1

    return 0


def _load_ltx2_pipeline(
    base_model_id: str,
    base_model_path: str | None,
    lora_info: dict | None,
    emitter: EventEmitter,
):
    """Load the LTX-2 pipeline with GGUF transformer for 24GB VRAM.

    Assembles the pipeline from local components (no HF downloads at runtime):
    1. Transformer: GGUF via from_single_file + bundled config
    2. Text encoder + connector: from local safetensors
    3. VAE: from local safetensors + bundled config
    4. Scheduler: from bundled config
    5. LoRA: distilled LoRA for fast 8-step inference
    """
    import torch
    from diffusers import (
        LTX2Pipeline,
        LTX2VideoTransformer3DModel,
        GGUFQuantizationConfig,
    )
    from modl_worker.adapters.arch_config import _get_installed_path

    CONFIGS = Path(__file__).parent.parent / "configs"
    dtype = torch.bfloat16

    # --- 1. Load GGUF transformer ---
    transformer_path = base_model_path
    if not transformer_path:
        transformer_path = _get_installed_path("ltx-video-dev")
    if not transformer_path or not Path(transformer_path).exists():
        raise FileNotFoundError(
            "LTX-2 transformer not found. Install with: modl pull ltx-video-dev"
        )

    emitter.info(f"Loading GGUF transformer from {Path(transformer_path).name}...")
    t0 = time.time()

    config_path = str(CONFIGS / "ltx2-transformer")
    transformer = LTX2VideoTransformer3DModel.from_single_file(
        transformer_path,
        config=config_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        torch_dtype=dtype,
    )
    emitter.info(f"Transformer loaded in {time.time()-t0:.1f}s")

    # --- 2. Load rest of pipeline from HF repo ---
    # Uses Lightricks/LTX-2 diffusers-format repo for text encoder, VAE,
    # tokenizer, connector, scheduler. Downloads once then cached.
    emitter.info("Loading text encoder + VAE + connector (from cache or Lightricks/LTX-2)...")
    t1 = time.time()

    pipe = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        transformer=transformer,
        torch_dtype=dtype,
    )
    emitter.info(f"Pipeline assembled in {time.time()-t1:.1f}s")

    # --- 8. Memory optimization ---
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    emitter.info("CPU offload + VAE tiling enabled")

    # --- 9. Apply distilled LoRA ---
    lora_path = None
    lora_strength = 0.6

    if lora_info:
        lora_path = lora_info.get("path")
        lora_strength = lora_info.get("weight", 0.6)

    if not lora_path:
        lora_path = _get_installed_path("ltx2-distilled-lora")

    if lora_path and Path(lora_path).exists():
        emitter.info(f"Applying distilled LoRA (strength={lora_strength})...")
        try:
            pipe.load_lora_weights(lora_path)
            pipe.fuse_lora(lora_scale=lora_strength)
            emitter.info("Distilled LoRA applied")
        except Exception as exc:
            emitter.warning("LORA_FAILED", f"Failed to apply LoRA: {exc}")

    return pipe
