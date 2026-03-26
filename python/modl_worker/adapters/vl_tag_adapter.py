"""VL Tag adapter — automatic image labeling using Qwen2.5-VL.

Generates comma-separated tags/labels for images using a vision-language model.

Reads a vl-tag job spec YAML containing:
  image_paths: list[str]    — paths to images
  model: str                — "qwen25-vl-3b" (default) or "qwen25-vl-7b"
  max_tags: int             — maximum number of tags (optional)
"""

import time
from pathlib import Path

from modl_worker.image_util import resolve_images
from modl_worker.protocol import EventEmitter


def run_vl_tag(config_path: Path, emitter: EventEmitter) -> int:
    """Run VL-based image tagging from a VlTagJobSpec YAML file."""
    import yaml
    from modl_worker.adapters.vl_common import load_qwen_vl, run_vl_inference

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"VL tag spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    image_paths = spec.get("image_paths", [])
    model_id = spec.get("model") or "qwen25-vl-3b"
    max_tags = spec.get("max_tags")

    images = resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to tag using {model_id}")
    emitter.job_started(config=str(config_path))

    try:
        model, processor = load_qwen_vl(emitter, model_id)
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load VL model: {exc}", recoverable=False)
        return 1

    prompt = "List the main objects and concepts in this image as comma-separated tags. Just the tags, nothing else."

    results = []
    errors = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="vl-tag", step=i, total_steps=total)

        try:
            t0 = time.time()
            response = run_vl_inference(model, processor, str(image_path), prompt, max_tokens=256)
            elapsed = time.time() - t0

            tags = [t.strip() for t in response.split(",") if t.strip()]
            if max_tags and len(tags) > max_tags:
                tags = tags[:max_tags]

            results.append({
                "image": str(image_path),
                "tags": tags,
            })

            emitter.info(f"[{i + 1}/{total}] {image_path.name} ({elapsed:.1f}s): {', '.join(tags[:5])}...")

        except Exception as exc:
            emitter.warning("VL_TAG_FAILED", f"Failed to tag {image_path.name}: {exc}")
            results.append({"image": str(image_path), "tags": []})
            errors += 1

    emitter.progress(stage="vl-tag", step=total, total_steps=total)

    emitter.result("vl_tagging", {
        "results": results,
        "images_processed": total,
        "errors": errors,
    })

    summary = f"Tagged {total - errors}/{total} images"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
