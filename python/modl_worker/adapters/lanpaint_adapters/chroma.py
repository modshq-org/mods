"""Chroma adapter for LanPaint.

Chroma uses packed latent sequences like Flux:
- Packed latents via _pack_latents / _unpack_latents
- Standard flow matching: x0 = x - flow_t * v
- Supports negative prompts (T5-XXL encoder)
- No reference image concatenation (not edit-style)
- timestep / 1000 normalization
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
        self._text_ids = None
        self._neg_text_ids = None
        self._attention_mask = None
        self._neg_attention_mask = None
        self._latent_image_ids = None
        self._height = None
        self._width = None

    def encode_prompt(self, prompt, negative_prompt):
        device = self.device
        # Move text encoder(s) to GPU
        if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
            self.pipe.text_encoder.to(device)
        if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None:
            self.pipe.text_encoder_2.to(device)

        with torch.no_grad():
            result = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                device=device,
            )
            # ChromaPipeline.encode_prompt returns (prompt_embeds, text_ids, attention_mask, ...)
            if isinstance(result, tuple):
                if len(result) >= 6:
                    # (prompt_embeds, text_ids, attention_mask, neg_embeds, neg_ids, neg_mask)
                    self._prompt_embeds = result[0].cpu()
                    self._text_ids = result[1].cpu()
                    self._attention_mask = result[2].cpu() if result[2] is not None else None
                    self._neg_prompt_embeds = result[3].cpu() if result[3] is not None else None
                    self._neg_text_ids = result[4].cpu() if result[4] is not None else None
                    self._neg_attention_mask = result[5].cpu() if result[5] is not None else None
                elif len(result) >= 3:
                    self._prompt_embeds = result[0].cpu()
                    self._text_ids = result[1].cpu()
                    self._attention_mask = result[2].cpu() if result[2] is not None else None
                elif len(result) == 2:
                    self._prompt_embeds = result[0].cpu()
                    self._neg_prompt_embeds = result[1].cpu() if result[1] is not None else None

        # Free text encoders
        if hasattr(self.pipe, 'text_encoder') and self.pipe.text_encoder is not None:
            self.pipe.text_encoder.to("cpu")
            del self.pipe.text_encoder
            self.pipe.text_encoder = None
        if hasattr(self.pipe, 'text_encoder_2') and self.pipe.text_encoder_2 is not None:
            self.pipe.text_encoder_2.to("cpu")
            del self.pipe.text_encoder_2
            self.pipe.text_encoder_2 = None
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

        # Scale + shift
        if hasattr(self.pipe.vae.config, "scaling_factor"):
            latent = latent * self.pipe.vae.config.scaling_factor
        if hasattr(self.pipe.vae.config, "shift_factor"):
            latent = latent - self.pipe.vae.config.shift_factor

        # Get spatial dims before packing
        self._height = img_tensor.shape[-2]
        self._width = img_tensor.shape[-1]

        # Pack latents
        num_ch = self.pipe.transformer.config.in_channels // 4
        latent_h, latent_w = latent.shape[-2], latent.shape[-1]

        # Prepare latent_image_ids
        self._latent_image_ids = self.pipe._prepare_latent_image_ids(
            1, latent_h // 2, latent_w // 2, device, torch.float32,
        ).cpu()

        # Pack: (B, C, H, W) → (B, L, C*4)
        packed = self.pipe._pack_latents(latent, 1, num_ch, latent_h, latent_w)

        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return packed.to(dtype=torch.float32)

    def mask_to_latent_space(self, mask, latent_shape):
        """Pixel mask → packed latent mask (1, L, 1)."""
        device = self.device
        # Latent spatial dims: latent_h/2, latent_w/2 (after packing)
        vae_scale = self.pipe.vae_scale_factor
        lat_h = self._height // vae_scale // 2
        lat_w = self._width // vae_scale // 2
        mask_latent = torch.nn.functional.interpolate(
            mask, size=(lat_h, lat_w), mode="nearest",
        ).to(device, torch.float32).reshape(1, -1, 1)
        return mask_latent

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
                timestep=t_tensor / 1000,
                img_ids=self._latent_image_ids.to(device),
                txt_ids=self._text_ids.to(device, model_dtype) if self._text_ids is not None else None,
                attention_mask=self._attention_mask.to(device) if self._attention_mask is not None else None,
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
                img_ids=self._latent_image_ids.to(device),
                txt_ids=self._neg_text_ids.to(device, model_dtype) if self._neg_text_ids is not None else None,
                attention_mask=self._neg_attention_mask.to(device) if self._neg_attention_mask is not None else None,
                return_dict=False,
            )[0]

        v_cfg = v_uncond.float() + guidance_scale * (v_cond.float() - v_uncond.float())
        v_big = v_uncond.float() + cfg_big * (v_cond.float() - v_uncond.float())

        x0 = x.float() - flow_t * v_cfg
        x0_big = x.float() - flow_t * v_big
        return x0, x0_big

    def decode_latents(self, latents):
        device = self.device
        self.pipe.vae.to(device)
        with torch.no_grad():
            # Unpack
            unpacked = self.pipe._unpack_latents(
                latents.to(device), self._height, self._width, self.pipe.vae_scale_factor,
            )
            # Reverse scale + shift
            decoded = (unpacked / self.pipe.vae.config.scaling_factor) + self.pipe.vae.config.shift_factor
            img = self.pipe.vae.decode(decoded.to(self.pipe.vae.dtype), return_dict=False)[0]
        result = self.pipe.image_processor.postprocess(img, output_type="pil")[0]
        self.pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return result
