"""Describe adapter — image captioning/description using Qwen2.5-VL.

Generates natural language descriptions of images using a vision-language model.

Reads a describe job spec YAML containing:
  image_paths: list[str]    — paths to images
  model: str                — "qwen25-vl-3b" (default) or "qwen25-vl-7b"
  detail: str               — "brief", "detailed", or "verbose" (default: "detailed")
"""

import time
from pathlib import Path

from modl_worker.image_util import resolve_images
from modl_worker.protocol import EventEmitter

PROMPTS = {
    "brief": "Describe this image in one sentence.",
    "detailed": "Describe this image in detail, including all objects, people, and the setting.",
    "verbose": "Provide a comprehensive description of this image. Include every visible object, person, text, color, spatial relationship, lighting, mood, and any notable details. Be thorough.",
}


def run_describe(config_path: Path, emitter: EventEmitter) -> int:
    """Run image description on images from a DescribeJobSpec YAML file."""
    import yaml
    from modl_worker.adapters.vl_common import load_qwen_vl, run_vl_inference

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Describe spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    detail = spec.get("detail", "detailed")
    model_id = spec.get("model") or "qwen25-vl-3b"

    images = resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to describe ({detail} mode, {model_id})")
    emitter.job_started(config=str(config_path))

    try:
        model, processor = load_qwen_vl(emitter, model_id)
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load VL model: {exc}", recoverable=False)
        return 1

    prompt = PROMPTS.get(detail, PROMPTS["detailed"])

    descriptions = []
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="describe", step=i, total_steps=total)

        try:
            t0 = time.time()
            caption = run_vl_inference(model, processor, str(image_path), prompt)
            elapsed = time.time() - t0

            descriptions.append({
                "image": str(image_path),
                "caption": caption,
            })

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s)")

        except Exception as exc:
            emitter.warning("DESCRIBE_FAILED", f"Failed to describe {image_path.name}: {exc}")
            descriptions.append({"image": str(image_path), "caption": "", "error": str(exc)})
            errors += 1

    emitter.progress(stage="describe", step=total, total_steps=total)

    emitter.result("description", {
        "descriptions": descriptions,
        "images_processed": total,
        "errors": errors,
    })

    summary = f"Described {total - errors}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
