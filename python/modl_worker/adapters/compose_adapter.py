"""Compose adapter — layer images onto a canvas with position/scale controls.

CPU-only operation using PIL. No GPU or model loading required.

Reads a ComposeJobSpec YAML containing:
  background: str           — path to background image (or "transparent"/"white"/"black")
  layers: list[dict]        — [{path, position, scale, opacity}]
  output_dir: str           — where to save the composite
  canvas_size: [w, h]       — optional explicit canvas size (defaults to background size)
"""

import time
from pathlib import Path

from PIL import Image

from modl_worker.image_util import load_image, save_and_emit_artifact
from modl_worker.protocol import EventEmitter


def _create_canvas(spec: dict) -> Image.Image:
    """Create the base canvas from background spec."""
    bg = spec.get("background", "transparent")
    canvas_size = spec.get("canvas_size")

    if bg in ("transparent", "white", "black"):
        if not canvas_size:
            raise ValueError("canvas_size required when background is a solid color")
        w, h = canvas_size
        if bg == "transparent":
            return Image.new("RGBA", (w, h), (0, 0, 0, 0))
        elif bg == "white":
            return Image.new("RGBA", (w, h), (255, 255, 255, 255))
        else:
            return Image.new("RGBA", (w, h), (0, 0, 0, 255))

    # Background is an image path
    canvas = load_image(bg, mode="RGBA")
    if canvas_size:
        w, h = canvas_size
        canvas = canvas.resize((w, h), Image.LANCZOS)
    return canvas


def _composite_layer(canvas: Image.Image, layer: dict) -> Image.Image:
    """Composite a single layer onto the canvas."""
    path = layer["path"]
    img = load_image(path, mode="RGBA")

    cw, ch = canvas.size
    lw, lh = img.size

    # Apply scale
    scale = layer.get("scale", 1.0)
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if scale != 1.0:
        new_w = max(1, int(lw * scale))
        new_h = max(1, int(lh * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        lw, lh = img.size

    # Position: always fractional 0.0–1.0, relative to canvas, placing center of layer
    pos = layer.get("position", [0.5, 0.5])
    px, py = pos

    # Convert fractional to pixel coordinates (center of the layer)
    px = int(px * cw)
    py = int(py * ch)

    # Position is center of layer — offset to top-left
    x = int(px - lw // 2)
    y = int(py - lh // 2)

    # Apply opacity
    opacity = layer.get("opacity", 1.0)
    if not (0.0 <= opacity <= 1.0):
        raise ValueError(f"opacity must be between 0.0 and 1.0, got {opacity}")
    if opacity < 1.0:
        alpha = img.getchannel("A")
        alpha = alpha.point(lambda a: int(a * opacity))
        img.putalpha(alpha)

    # Paste with alpha compositing
    canvas.paste(img, (x, y), img)
    return canvas


def run_compose(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run image composition from a ComposeJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Compose spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    layers = spec.get("layers", [])
    output_dir = spec.get("output_dir", ".")

    if not layers:
        emitter.error("NO_LAYERS", "No layers specified", recoverable=False)
        return 2

    # Validate layer paths exist
    for i, layer in enumerate(layers):
        p = Path(layer["path"])
        if not p.exists():
            emitter.error("LAYER_NOT_FOUND", f"Layer {i} not found: {p}", recoverable=False)
            return 2

    bg = spec.get("background", "transparent")
    if bg not in ("transparent", "white", "black") and not Path(bg).exists():
        emitter.error("BG_NOT_FOUND", f"Background image not found: {bg}", recoverable=False)
        return 2

    emitter.info(f"Compositing {len(layers)} layer(s)")
    emitter.job_started(config=str(config_path))

    t0 = time.time()

    try:
        canvas = _create_canvas(spec)

        for i, layer in enumerate(layers):
            emitter.progress(stage="compose", step=i, total_steps=len(layers))
            canvas = _composite_layer(canvas, layer)

        emitter.progress(stage="compose", step=len(layers), total_steps=len(layers))
        elapsed = time.time() - t0

        # Save output
        metadata = {
            "type": "compose",
            "background": bg,
            "layers": len(layers),
        }

        filepath = save_and_emit_artifact(
            canvas, output_dir, emitter,
            metadata=metadata, stage="compose", elapsed=elapsed,
        )

        emitter.result("compose", {
            "output": filepath,
            "output_dir": output_dir,
            "canvas_size": list(canvas.size),
            "layers": len(layers),
        })

        emitter.completed(f"Composite saved ({elapsed:.1f}s)")
        return 0

    except Exception as exc:
        emitter.error("COMPOSE_FAILED", f"Composition failed: {exc}", recoverable=False)
        return 1
