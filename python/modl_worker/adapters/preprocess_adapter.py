"""Preprocess adapter — extract control images (canny, depth, pose, softedge, scribble, lineart, normal).

Reads a PreprocessJobSpec YAML containing:
  image_paths: list[str]  — paths to images
  method: str             — canny, depth, pose, softedge, scribble, lineart, lineart_coarse, normal
  output_dir: str|None    — override output location (default: same dir as input)
  canny_low: int          — low threshold for Canny (default: 100)
  canny_high: int         — high threshold for Canny (default: 200)
  depth_model: str        — small, base (default: small)
  scribble_threshold: int — binary threshold for scribble (default: 128)
  include_hands: bool     — include hand keypoints in pose (default: true)
  include_face: bool      — include face landmarks in pose (default: true)
"""

import time
from pathlib import Path

import numpy as np

from modl_worker.protocol import EventEmitter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _resolve_images(image_paths: list[str]) -> list[Path]:
    """Expand directories and filter to valid image files."""
    result = []
    for p_str in image_paths:
        p = Path(p_str)
        if p.is_dir():
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
                    result.append(f)
        elif p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            result.append(p)
    return result


def _output_path(image_path: Path, method: str, output_dir: str | None) -> Path:
    """Compute output path: {stem}_{method}.png in output_dir or same directory."""
    # Normalize method name for filename (lineart_coarse -> lineart)
    method_suffix = method.replace("_coarse", "")
    name = f"{image_path.stem}_{method_suffix}.png"
    if output_dir:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / name
    return image_path.parent / name


# ---------------------------------------------------------------------------
# Canny — pure OpenCV, no model
# ---------------------------------------------------------------------------

def _preprocess_canny(img_np: np.ndarray, low: int, high: int) -> np.ndarray:
    """Canny edge detection. Returns a grayscale edge map."""
    import cv2
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low, high)
    return edges


# ---------------------------------------------------------------------------
# Depth — Depth Anything V2
# ---------------------------------------------------------------------------

_depth_model_cache: dict = {}


def _preprocess_depth(img_np: np.ndarray, model_variant: str, emitter: EventEmitter) -> np.ndarray:
    """Estimate depth using Depth Anything V2. Returns a grayscale depth map."""
    import torch

    cache_key = f"depth_{model_variant}"
    if cache_key in _depth_model_cache:
        model, transform = _depth_model_cache[cache_key]
    else:
        emitter.info(f"Loading Depth Anything V2 ({model_variant})...")

        from transformers import AutoModelForDepthEstimation, AutoImageProcessor

        model_ids = {
            "small": "depth-anything/Depth-Anything-V2-Small-hf",
            "base": "depth-anything/Depth-Anything-V2-Base-hf",
            "large": "depth-anything/Depth-Anything-V2-Large-hf",
        }
        model_id = model_ids.get(model_variant, model_ids["small"])

        transform = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForDepthEstimation.from_pretrained(model_id)

        from modl_worker.device import get_device
        device = get_device()
        model = model.to(device).eval()
        _depth_model_cache[cache_key] = (model, transform)
        emitter.info("Depth model loaded")

    from PIL import Image
    import torch

    device = next(model.parameters()).device
    img_pil = Image.fromarray(img_np)
    h, w = img_np.shape[:2]

    inputs = transform(images=img_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth

    # Interpolate to original size
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1), size=(h, w), mode="bicubic", align_corners=False,
    ).squeeze()

    # Normalize to 0-255
    depth_np = depth.cpu().numpy()
    depth_np = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min() + 1e-8) * 255
    return depth_np.astype(np.uint8)


# ---------------------------------------------------------------------------
# Pose — DWPose (via controlnet_aux)
# ---------------------------------------------------------------------------

_pose_processor_cache: dict = {}


def _preprocess_pose(img_np: np.ndarray, include_hands: bool, include_face: bool,
                     emitter: EventEmitter) -> np.ndarray:
    """Extract pose skeleton using DWPose via controlnet_aux."""
    cache_key = "dwpose"
    if cache_key in _pose_processor_cache:
        processor = _pose_processor_cache[cache_key]
    else:
        emitter.info("Loading DWPose processor...")
        from controlnet_aux import DWposeDetector
        processor = DWposeDetector.from_pretrained("yzd-v/DWPose")
        _pose_processor_cache[cache_key] = processor
        emitter.info("DWPose loaded")

    from PIL import Image

    img_pil = Image.fromarray(img_np)
    result = processor(
        img_pil,
        include_hand=include_hands,
        include_face=include_face,
        output_type="np",
    )

    if isinstance(result, np.ndarray):
        return result
    # controlnet_aux returns PIL sometimes
    return np.array(result)


# ---------------------------------------------------------------------------
# Softedge — HED (via controlnet_aux)
# ---------------------------------------------------------------------------

_hed_processor_cache: dict = {}


def _preprocess_softedge(img_np: np.ndarray, emitter: EventEmitter) -> np.ndarray:
    """Extract soft edges using HED."""
    cache_key = "hed"
    if cache_key in _hed_processor_cache:
        processor = _hed_processor_cache[cache_key]
    else:
        emitter.info("Loading HED processor...")
        from controlnet_aux import HEDdetector
        processor = HEDdetector.from_pretrained("lllyasviel/Annotators")
        _hed_processor_cache[cache_key] = processor
        emitter.info("HED loaded")

    from PIL import Image

    img_pil = Image.fromarray(img_np)
    result = processor(img_pil, output_type="np")
    if isinstance(result, np.ndarray):
        return result
    return np.array(result)


# ---------------------------------------------------------------------------
# Scribble — HED + binary threshold
# ---------------------------------------------------------------------------

def _preprocess_scribble(img_np: np.ndarray, threshold: int, emitter: EventEmitter) -> np.ndarray:
    """Extract scribble lines (HED + binary threshold)."""
    softedge = _preprocess_softedge(img_np, emitter)
    # Convert to grayscale if needed
    if softedge.ndim == 3:
        gray = np.mean(softedge, axis=2).astype(np.uint8)
    else:
        gray = softedge
    # Invert so lines are white on black, then threshold
    _, binary = __import__("cv2").threshold(255 - gray, threshold, 255, __import__("cv2").THRESH_BINARY)
    return binary


# ---------------------------------------------------------------------------
# Lineart — via controlnet_aux
# ---------------------------------------------------------------------------

_lineart_processor_cache: dict = {}


def _preprocess_lineart(img_np: np.ndarray, coarse: bool, emitter: EventEmitter) -> np.ndarray:
    """Extract line art."""
    cache_key = f"lineart_{'coarse' if coarse else 'standard'}"
    if cache_key in _lineart_processor_cache:
        processor = _lineart_processor_cache[cache_key]
    else:
        emitter.info("Loading Lineart processor...")
        from controlnet_aux import LineartDetector
        processor = LineartDetector.from_pretrained("lllyasviel/Annotators")
        _lineart_processor_cache[cache_key] = processor
        emitter.info("Lineart loaded")

    from PIL import Image

    img_pil = Image.fromarray(img_np)
    result = processor(img_pil, coarse=coarse, output_type="np")
    if isinstance(result, np.ndarray):
        return result
    return np.array(result)


# ---------------------------------------------------------------------------
# Normal — derive from depth map
# ---------------------------------------------------------------------------

def _preprocess_normal(img_np: np.ndarray, model_variant: str, emitter: EventEmitter) -> np.ndarray:
    """Derive normal map from depth estimation."""
    depth = _preprocess_depth(img_np, model_variant, emitter).astype(np.float32)

    # Sobel gradients for normals
    import cv2
    grad_x = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)

    # Build normal map: [-1,1] per channel → [0,255]
    normal = np.zeros((*depth.shape, 3), dtype=np.float32)
    normal[..., 0] = -grad_x  # R = X
    normal[..., 1] = -grad_y  # G = Y
    normal[..., 2] = 1.0      # B = Z

    # Normalize per pixel
    norm = np.linalg.norm(normal, axis=2, keepdims=True)
    normal = normal / (norm + 1e-8)

    # Map from [-1, 1] to [0, 255]
    normal_rgb = ((normal + 1) / 2 * 255).astype(np.uint8)
    return normal_rgb


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

METHODS = {
    "canny", "depth", "pose", "softedge", "scribble",
    "lineart", "lineart_coarse", "normal",
}


def run_preprocess(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run preprocessing from a PreprocessJobSpec YAML file."""
    import yaml

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Preprocess spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    method = spec.get("method", "canny")
    if method not in METHODS:
        emitter.error(
            "INVALID_METHOD",
            f"Unknown preprocess method: {method}. Options: {', '.join(sorted(METHODS))}",
            recoverable=False,
        )
        return 2

    image_paths = spec.get("image_paths", [])
    output_dir = spec.get("output_dir")
    canny_low = spec.get("canny_low", 100)
    canny_high = spec.get("canny_high", 200)
    depth_model = spec.get("depth_model", "small")
    scribble_threshold = spec.get("scribble_threshold", 128)
    include_hands = spec.get("include_hands", True)
    include_face = spec.get("include_face", True)

    images = _resolve_images(image_paths)
    if not images:
        emitter.error("NO_IMAGES", "No valid images found to preprocess", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} image(s) to preprocess (method: {method})")
    emitter.job_started(config=str(config_path))

    from PIL import Image
    from modl_worker.image_util import load_image

    processed = 0
    errors = 0
    outputs_list = []

    for i, image_path in enumerate(images):
        emitter.progress(stage="preprocess", step=i, total_steps=total)

        try:
            t0 = time.time()
            img_pil = load_image(image_path)
            img_np = np.array(img_pil)

            # Dispatch to method
            if method == "canny":
                result_np = _preprocess_canny(img_np, canny_low, canny_high)
            elif method == "depth":
                result_np = _preprocess_depth(img_np, depth_model, emitter)
            elif method == "pose":
                result_np = _preprocess_pose(img_np, include_hands, include_face, emitter)
            elif method == "softedge":
                result_np = _preprocess_softedge(img_np, emitter)
            elif method == "scribble":
                result_np = _preprocess_scribble(img_np, scribble_threshold, emitter)
            elif method in ("lineart", "lineart_coarse"):
                result_np = _preprocess_lineart(img_np, coarse=(method == "lineart_coarse"), emitter=emitter)
            elif method == "normal":
                result_np = _preprocess_normal(img_np, depth_model, emitter)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Save output
            out_path = _output_path(image_path, method, output_dir)
            if result_np.ndim == 2:
                # Grayscale
                out_img = Image.fromarray(result_np, mode="L")
            else:
                out_img = Image.fromarray(result_np)

            # Resize to match input dimensions if needed
            if out_img.size != img_pil.size:
                out_img = out_img.resize(img_pil.size, Image.LANCZOS)

            out_img.save(str(out_path))

            elapsed = time.time() - t0
            emitter.info(f"[{i + 1}/{total}] {image_path.name} → {out_path.name} ({elapsed:.1f}s)")
            emitter.artifact(path=str(out_path))

            outputs_list.append({
                "input": str(image_path),
                "output": str(out_path),
                "method": method,
                "resolution": [img_pil.width, img_pil.height],
            })
            processed += 1

        except Exception as exc:
            emitter.warning("PREPROCESS_FAILED", f"Failed to preprocess {image_path.name}: {exc}")
            errors += 1

    emitter.progress(stage="preprocess", step=total, total_steps=total)

    emitter.result("preprocess", {
        "processed": processed,
        "errors": errors,
        "method": method,
        "outputs": outputs_list,
    })

    summary = f"Preprocessed {processed}/{total} images (method: {method})"
    if errors > 0:
        summary += f" ({errors} error(s))"
    emitter.completed(summary)

    return 0 if errors == 0 else 1
