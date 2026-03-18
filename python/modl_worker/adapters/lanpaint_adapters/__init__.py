"""Per-model adapters for LanPaint inpainting."""

from .base import ModelAdapter
from .z_image import ZImageAdapter
from .flux_klein import FluxKleinAdapter
from .chroma import ChromaAdapter

__all__ = ["ModelAdapter", "ZImageAdapter", "FluxKleinAdapter", "ChromaAdapter"]
