"""Device resolution for modl worker.

Reads MODL_DEVICE env var (set by the Rust CLI based on detected hardware),
with fallback to PyTorch runtime detection.

Usage:
    from modl_worker.device import get_device, get_generator_device
    pipe.to(get_device())
    generator = torch.Generator(device=get_generator_device())
"""

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def get_device() -> str:
    """Return the torch device string: 'cuda', 'mps', or 'cpu'."""
    env = os.environ.get("MODL_DEVICE", "").lower()
    if env in ("cuda", "mps", "cpu"):
        return env
    # Auto-detect
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_generator_device() -> str:
    """Return the device for torch.Generator.

    MPS has quirks with Generator — some ops require CPU generator even when
    running on MPS. Use 'cpu' for MPS, actual device otherwise.
    """
    dev = get_device()
    if dev == "mps":
        return "cpu"
    return dev


def empty_cache():
    """Free GPU cache for the active device."""
    dev = get_device()
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()
