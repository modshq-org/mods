"""ModelAdapter — abstract base class for LanPaint model adapters.

Each concrete adapter (ZImageAdapter, FluxKleinAdapter, ...) implements
the model-specific operations. The generic LanPaint orchestrator delegates
all model-specific work to the adapter.

Adapted from LanPaint-diffusers (charrywhite/LanPaint-diffusers).
"""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from PIL import Image

from modl_worker.protocol import EventEmitter


class ModelAdapter(ABC):
    """Abstract base for adapting a diffusers pipeline to LanPaint."""

    def __init__(self, pipe, emitter: EventEmitter):
        self.pipe = pipe
        self.emitter = emitter

    @property
    def device(self) -> torch.device:
        from modl_worker.device import get_device
        return torch.device(get_device())

    @property
    def dtype(self) -> torch.dtype:
        model = getattr(self.pipe, "transformer", None) or getattr(self.pipe, "unet", None)
        return model.dtype if model else torch.bfloat16

    @property
    def scheduler(self):
        return self.pipe.scheduler

    @property
    def distilled(self) -> bool:
        """Whether this model is distilled (LanPaint quality degrades)."""
        return False

    def noise_scaling(self, sigma, noise, latent_image):
        """CONST flow-matching: x_t = (1-t)*x0 + t*noise."""
        return (1.0 - sigma) * latent_image + sigma * noise

    @abstractmethod
    def encode_prompt(self, prompt: str, negative_prompt: str) -> None:
        """Encode prompts and store internally. Called with text encoder on GPU."""
        ...

    @abstractmethod
    def encode_image(self, img_tensor: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        """VAE-encode image. Returns clean latent in float32. Called with VAE on GPU."""
        ...

    @abstractmethod
    def prepare_timesteps(self, num_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Set up scheduler. Returns (timesteps, flow_ts) both on device."""
        ...

    @abstractmethod
    def predict_x0(self, x: torch.Tensor, flow_t: float, guidance_scale: float, cfg_big: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict x0 from noisy latent. Returns (x0_std, x0_big). Called with transformer on GPU."""
        ...

    @abstractmethod
    def decode_latents(self, latents: torch.Tensor) -> Image.Image:
        """Decode latents to PIL image. Called with VAE on GPU."""
        ...

    def mask_to_latent_space(self, mask: torch.Tensor, latent_shape: tuple) -> torch.Tensor:
        """Downsample pixel mask to latent space. Default: nearest interpolation."""
        latent_h, latent_w = latent_shape[-2], latent_shape[-1]
        return torch.nn.functional.interpolate(
            mask, size=(latent_h, latent_w), mode="nearest"
        )
