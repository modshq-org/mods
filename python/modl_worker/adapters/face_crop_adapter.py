"""Face crop adapter — detect faces and create close-up crops for character LoRA datasets.

Detects faces using InsightFace, crops with generous padding to include
head + shoulders, resizes to training resolution, and generates captions
for the crops.

This dramatically improves character LoRA training quality by giving the
model more face pixels to learn identity from.

Reads a face_crop job spec YAML containing:
  dataset_path: str         — path to the dataset directory
  resolution: int           — target crop resolution (default 1024)
  padding: float            — bbox expansion multiplier (default 1.8)
  trigger_word: str         — trigger word to use in generated captions
  class_word: str           — class word (e.g. "man", "woman", "dog")
  caption_prefix: str       — optional caption prefix override
"""

import time
from pathlib import Path

from modl_worker.protocol import EventEmitter

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _find_images(dataset_path: Path) -> list[Path]:
    """Find all image files in the dataset directory."""
    images = []
    for f in sorted(dataset_path.iterdir()):
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS:
            # Skip files that are already face crops
            if "_facecrop_" in f.stem:
                continue
            images.append(f)
    return images


def _expand_bbox(bbox: list[float], img_w: int, img_h: int, padding: float) -> tuple[int, int, int, int]:
    """Expand a face bounding box by a padding multiplier and make it square.

    The padding multiplier controls how much context around the face to include:
    - 1.0 = tight face crop
    - 1.5 = face + some hair/neck
    - 1.8 = head + shoulders (recommended for character LoRAs)
    - 2.5 = upper body

    Returns (x1, y1, x2, y2) clamped to image bounds.
    """
    x1, y1, x2, y2 = bbox
    face_w = x2 - x1
    face_h = y2 - y1

    # Use the larger dimension for a square crop
    size = max(face_w, face_h) * padding

    # Center the crop on the face center, biased slightly upward
    # (faces are usually in the upper portion of a good portrait)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2 - face_h * 0.05  # slight upward bias

    half = size / 2
    crop_x1 = int(max(0, cx - half))
    crop_y1 = int(max(0, cy - half))
    crop_x2 = int(min(img_w, cx + half))
    crop_y2 = int(min(img_h, cy + half))

    return crop_x1, crop_y1, crop_x2, crop_y2


def _read_caption(image_path: Path) -> str | None:
    """Read the .txt caption file paired with an image."""
    txt_path = image_path.with_suffix(".txt")
    if txt_path.exists():
        return txt_path.read_text().strip()
    return None


def _make_crop_caption(
    original_caption: str | None,
    trigger_word: str,
    class_word: str | None,
) -> str:
    """Generate a caption for a face crop from the original caption.

    Strategy: take the original caption and transform it to describe a close-up.
    If no original caption, generate a simple one from trigger + class word.
    """
    subject = f"{trigger_word} {class_word}" if class_word else trigger_word

    if not original_caption:
        return f"a close-up photo of {subject}"

    # Replace common photo-type prefixes with close-up variants
    caption = original_caption
    # Already a close-up? Keep as-is.
    if caption.lower().startswith("a close-up"):
        return caption
    replacements = [
        ("a full body photo of", "a close-up photo of"),
        ("a full-body photo of", "a close-up photo of"),
        ("a photo of", "a close-up photo of"),
        ("a portrait of", "a close-up portrait of"),
        ("a headshot of", "a close-up headshot of"),
    ]
    replaced = False
    for old, new in replacements:
        if caption.lower().startswith(old):
            caption = new + caption[len(old):]
            replaced = True
            break

    if not replaced:
        caption = f"a close-up photo of {subject}, " + caption

    # Remove scene-level descriptions that won't be visible in a face crop
    # (these are after the last comma in many captions)
    remove_phrases = [
        "crowd in background",
        "audience in background",
        "blurred crowd behind",
        "full body",
        "full-body",
        "seen from above",
        "shot from above",
    ]
    for phrase in remove_phrases:
        caption = caption.replace(phrase, "").replace("  ", " ")

    # Clean up trailing commas and whitespace
    caption = caption.strip().rstrip(",").strip()

    return caption


def run_face_crop(config_path: Path, emitter: EventEmitter, model_cache: dict | None = None) -> int:
    """Run face detection + cropping on a dataset."""
    import yaml
    import cv2
    from PIL import Image

    if not config_path.exists():
        emitter.error("SPEC_NOT_FOUND", f"Spec not found: {config_path}", recoverable=False)
        return 2

    try:
        with open(config_path) as f:
            spec = yaml.safe_load(f)
    except Exception as exc:
        emitter.error("SPEC_PARSE_ERROR", str(exc), recoverable=False)
        return 2

    dataset_path = Path(spec["dataset_path"])
    resolution = spec.get("resolution", 1024)
    padding = spec.get("padding", 1.8)
    trigger_word = spec.get("trigger_word", "")
    class_word = spec.get("class_word", "")

    if not dataset_path.exists():
        emitter.error("DATASET_NOT_FOUND", f"Dataset not found: {dataset_path}", recoverable=False)
        return 2

    images = _find_images(dataset_path)
    if not images:
        emitter.error("NO_IMAGES", "No images found in dataset", recoverable=False)
        return 2

    total = len(images)
    emitter.info(f"Found {total} images to process (padding={padding}x, resolution={resolution}px)")
    emitter.job_started(config=str(config_path))

    # Load InsightFace
    try:
        if model_cache is not None and "insightface_app" in model_cache:
            app = model_cache["insightface_app"]
            emitter.info("Using cached InsightFace model")
        else:
            from insightface.app import FaceAnalysis
            emitter.info("Loading InsightFace buffalo_l model...")
            app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            if model_cache is not None:
                model_cache["insightface_app"] = app
    except Exception as exc:
        emitter.error("MODEL_LOAD_FAILED", f"Failed to load InsightFace: {exc}", recoverable=False)
        return 1

    emitter.info("Model loaded, detecting faces and cropping...")

    crops_created = 0
    images_with_faces = 0
    skipped = 0

    for i, image_path in enumerate(images):
        emitter.progress(stage="face_crop", step=i, total_steps=total)

        try:
            t0 = time.time()

            # Detect faces
            img_cv = cv2.imread(str(image_path))
            if img_cv is None:
                emitter.warning("READ_FAILED", f"Could not read: {image_path.name}")
                continue

            faces = app.get(img_cv)
            if not faces:
                emitter.info(f"[{i+1}/{total}] {image_path.name}: no faces detected, skipping")
                continue

            images_with_faces += 1
            img_pil = Image.open(image_path)
            img_w, img_h = img_pil.size
            original_caption = _read_caption(image_path)

            # Sort faces by bbox area (largest first — primary subject)
            faces_sorted = sorted(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

            # Only crop the largest face (primary subject)
            face = faces_sorted[0]
            confidence = float(face.det_score)

            # Skip low-confidence detections
            if confidence < 0.5:
                emitter.info(f"[{i+1}/{total}] {image_path.name}: face confidence too low ({confidence:.2f}), skipping")
                continue

            # Compute crop region
            crop_x1, crop_y1, crop_x2, crop_y2 = _expand_bbox(
                face.bbox, img_w, img_h, padding
            )
            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1

            # Skip if the crop would be almost the same as the original
            # (face already fills most of the image)
            face_area_ratio = (crop_w * crop_h) / (img_w * img_h)
            if face_area_ratio > 0.85:
                emitter.info(f"[{i+1}/{total}] {image_path.name}: face already fills frame, skipping crop")
                skipped += 1
                continue

            # Skip if crop would be tiny (face too small / too far away)
            if crop_w < 128 or crop_h < 128:
                emitter.info(f"[{i+1}/{total}] {image_path.name}: face too small ({crop_w}x{crop_h}px), skipping")
                skipped += 1
                continue

            # Crop and resize
            cropped = img_pil.crop((crop_x1, crop_y1, crop_x2, crop_y2))
            cropped = cropped.resize((resolution, resolution), Image.LANCZOS)

            # Save crop
            crop_filename = f"{image_path.stem}_facecrop_0{image_path.suffix}"
            crop_path = dataset_path / crop_filename
            cropped.save(str(crop_path), quality=95)

            # Generate and save caption
            crop_caption = _make_crop_caption(original_caption, trigger_word, class_word)
            caption_path = crop_path.with_suffix(".txt")
            caption_path.write_text(crop_caption)

            elapsed = time.time() - t0
            crops_created += 1
            emitter.info(
                f"[{i+1}/{total}] {image_path.name} ({elapsed:.1f}s): "
                f"cropped {crop_w}x{crop_h}px → {resolution}x{resolution}px"
            )

        except Exception as exc:
            emitter.warning("CROP_FAILED", f"Failed to process {image_path.name}: {exc}")

    emitter.progress(stage="face_crop", step=total, total_steps=total)

    summary = (
        f"Created {crops_created} face crop(s) from {images_with_faces} images with faces "
        f"({total} images scanned, {skipped} skipped)"
    )
    emitter.result("face_crop", {
        "crops_created": crops_created,
        "images_with_faces": images_with_faces,
        "images_scanned": total,
        "skipped": skipped,
    })
    emitter.completed(summary)
    return 0
