"""Chroma adapter for LanPaint.

Similar to Flux but simpler:
- Spatial latents (B, C, H, W), not packed
- Standard flow matching: x0 = x - flow_t * v
- Supports negative prompts (separate cond/uncond passes)
- No reference image concatenation
- Uses T5-XXL text encoder
"""

from typing import Tuple

import torch
from PIL import Image

from .base import ModelAdapter


class ChromaAdapter(ModelAdapter):
    """LanPaint adapter for Chroma."""

    def __init__(self, pipe, emitter):
        super().__init__(pipe, emitter)
        self._prompt_embeds = None
        self._neg_prompt_embeds = None
        self._txt_ids = None
        self._neg_txt_ids = None
        self._img_ids = None

    def encode_prompt(self, prompt, negative_prompt):
        device = self.device
        # Chroma uses T5-XXL (text_encoder_2 or text_encoder)
        te = getattr(self.pipe, 'text_encoder_2', None) or self.pipe.text_encoder
        te.to(device)

        with torch.no_grad():
            result = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                device=device,
            )
            if len(result) == 4:
                self._prompt_embeds, self._neg_prompt_embeds, _, _ = result
            elif len(result) == 2:
                self._prompt_embeds, self._neg_prompt_embeds = result
            else:
                self._prompt_embeds = result[0]
                self._neg_prompt_embeds = None

        # Move to CPU
        self._prompt_embeds = self._prompt_embeds.cpu()
        if self._neg_prompt_embeds is not None:
            self._neg_prompt_embeds = self._neg_prompt_embeds.cpu()

        # Prepare txt_ids
        seq_len = self._prompt_embeds.shape[1]
        self._txt_ids = torch.zeros(seq_len, 3, dtype=torch.float32)
        if self._neg_prompt_embeds is not None:
            neg_seq_len = self._neg_prompt_embeds.shape[1]
            self._neg_txt_ids = torch.zeros(neg_seq_len, 3, dtype=torch.float32)

        te.to("cpu")
        torch.cuda.empty_cache()
        import gc; gc.collect()
        self.emitter.info(f"  Text encoder freed: {torch.cuda.memory_allocated()/1e9:.1f}GB used")

    def encode_image(self, img_tensor, generator):
        device = self.device
        self.pipe.vae.to(device)
        img_tensor = img_tensor.to(device=device, dtype=self.pipe.vae.dtype)

        with torch.no_grad():
            dist = self.pipe.vae.encode(img_tensor)
            latent = dist.latent_dist.mode() if hasattr(dist, "latent_dist") else dist.mode()

        # Scale (same VAE as Flux)
        if hasattr(self.pipe.vae.config, "scaling_factor"):
            latent = latent * self.pipe.vae.config.scaling_factor
        if hasattr(self.pipe.vae.config, "shift_factor"):
            latent = latent - self.pipe.vae.config.shift_factor

        # Prepare img_ids
        h, w = latent.shape[-2], latent.shape[-1]
        self._img_ids = self.pipe._prepare_latent_image_ids(1, h, w, device, torch.float32)

        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return latent.to(dtype=torch.float32)

    def prepare_timesteps(self, num_steps):
        device = self.device
        self.pipe.scheduler.set_timesteps(num_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps
        flow_ts = self.pipe.scheduler.sigmas[:-1].to(device=device, dtype=torch.float32)
        return timesteps[:len(flow_ts)], flow_ts

    def predict_x0(self, x, flow_t, guidance_scale, cfg_big):
        device = self.device
        model_dtype = self.dtype
        batch = x.shape[0]

        t_tensor = torch.full((batch,), flow_t, device=device, dtype=model_dtype)

        # Conditional pass
        with torch.no_grad():
            v_cond = self.pipe.transformer(
                hidden_states=x.to(model_dtype),
                encoder_hidden_states=self._prompt_embeds.to(device, model_dtype),
                timestep=t_tensor / 1000,  # Chroma expects normalized timestep
                img_ids=self._img_ids.to(device) if self._img_ids is not None else None,
                txt_ids=self._txt_ids.to(device, model_dtype),
                return_dict=False,
            )[0]

        if guidance_scale <= 1.0 or self._neg_prompt_embeds is None:
            x0 = x.float() - flow_t * v_cond.float()
            return x0, x0

        # Unconditional pass
        with torch.no_grad():
            v_uncond = self.pipe.transformer(
                hidden_states=x.to(model_dtype),
                encoder_hidden_states=self._neg_prompt_embeds.to(device, model_dtype),
                timestep=t_tensor / 1000,
                img_ids=self._img_ids.to(device) if self._img_ids is not None else None,
                txt_ids=self._neg_txt_ids.to(device, model_dtype) if self._neg_txt_ids is not None else None,
                return_dict=False,
            )[0]

        # Dual CFG + x0 (standard flow: x0 = x - t*v)
        v_cfg = v_uncond.float() + guidance_scale * (v_cond.float() - v_uncond.float())
        v_big = v_uncond.float() + cfg_big * (v_cond.float() - v_uncond.float())

        x0 = x.float() - flow_t * v_cfg
        x0_big = x.float() - flow_t * v_big
        return x0, x0_big

    def decode_latents(self, latents):
        device = self.device
        self.pipe.vae.to(device)
        with torch.no_grad():
            decoded = latents.to(self.pipe.vae.dtype)
            if hasattr(self.pipe.vae.config, "shift_factor"):
                decoded = (decoded / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            else:
                decoded = decoded / self.pipe.vae.config.scaling_factor
            img = self.pipe.vae.decode(decoded, return_dict=False)[0]
        result = self.pipe.image_processor.postprocess(img, output_type="pil")[0]
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return result
