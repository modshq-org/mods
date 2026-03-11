"""Shared image loading utilities."""

from pathlib import Path
from PIL import Image, ImageOps


def load_image(path: str | Path, mode: str = "RGB") -> Image.Image:
    """Open an image, apply EXIF orientation, and convert to the given mode.

    Phone cameras typically store images in landscape with an EXIF orientation
    tag. ``ImageOps.exif_transpose`` physically rotates the pixels to match the
    intended orientation and strips the tag so downstream code sees the correct
    geometry.
    """
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if mode and img.mode != mode:
        img = img.convert(mode)
    return img
