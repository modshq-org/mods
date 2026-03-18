"""Qwen Image adapter for LanPaint.

Key conventions:
- Packed 2x2 patches: (B, L, C*4=64)
- Edit-style: reference image concatenated at every forward pass
- Deferred prompt encoding (VL encoder needs the image)
- Per-channel mean/std VAE normalization
- Norm rescaling on CFG output
- Cache-context based CFG
- Standard flow matching: x0 = x - flow_t * v

Reference: LanPaint-diffusers QwenAdapter (charrywhite/LanPaint-diffusers).
"""

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from .base import ModelAdapter


def _retrieve_latents(encoder_output, generator=None):
    if hasattr(encoder_output, "latent_dist"):
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents")


class QwenImageAdapter(ModelAdapter):
    """LanPaint adapter for Qwen Image Edit."""

    def __init__(self, pipe, emitter):
        super().__init__(pipe, emitter)
        self._prompt_str = None
        self._neg_prompt_str = None
        self._prompt_embeds = None
        self._prompt_embeds_mask = None
        self._neg_prompt_embeds = None
        self._neg_prompt_embeds_mask = None
        self._image_latents_packed = None
        self._img_shapes = None
        self._latent_height = 0
        self._latent_width = 0
        self._pixel_height = 0
        self._pixel_width = 0
        self._init_tensor = None  # stored for deferred prompt encoding

    def encode_prompt(self, prompt, negative_prompt):
        """Defer encoding — Qwen VL encoder needs the image."""
        self._prompt_str = prompt
        self._neg_prompt_str = negative_prompt

    def encode_image(self, img_tensor, generator):
        """VAE-encode + deferred prompt encoding (VL encoder needs image)."""
        device = self.device
        pipe = self.pipe
        model_dtype = self.dtype

        self._init_tensor = img_tensor
        height = img_tensor.shape[-2]
        width = img_tensor.shape[-1]
        self._pixel_height = height
        self._pixel_width = width

        vae_sf = pipe.vae_scale_factor
        self._latent_height = 2 * (int(height) // (vae_sf * 2))
        self._latent_width = 2 * (int(width) // (vae_sf * 2))

        # --- 1. Encode prompts with VL encoder (needs condition image) ---
        # Reconstruct PIL for VL encoder
        img_np = img_tensor[0].detach().float().cpu()
        img_np = ((img_np / 2.0) + 0.5).clamp(0.0, 1.0)
        img_np = img_np.permute(1, 2, 0).numpy()
        condition_pil = Image.fromarray((img_np * 255).astype(np.uint8))

        # Resize for VL encoder
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            CONDITION_IMAGE_SIZE, calculate_dimensions,
        )
        img_w, img_h = condition_pil.size
        cond_w, cond_h = calculate_dimensions(CONDITION_IMAGE_SIZE, img_w / img_h)
        condition_image = pipe.image_processor.resize(condition_pil, cond_h, cond_w)

        # Move text encoder to GPU for encoding
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            pipe.text_encoder.to(device)

        with torch.no_grad():
            self._prompt_embeds, self._prompt_embeds_mask = pipe.encode_prompt(
                prompt=self._prompt_str,
                image=[condition_image],
                device=device,
            )
            neg_str = self._neg_prompt_str or ""
            self._neg_prompt_embeds, self._neg_prompt_embeds_mask = pipe.encode_prompt(
                prompt=neg_str,
                image=[condition_image],
                device=device,
            )

        # Move to CPU
        self._prompt_embeds = self._prompt_embeds.cpu()
        self._prompt_embeds_mask = self._prompt_embeds_mask.cpu()
        self._neg_prompt_embeds = self._neg_prompt_embeds.cpu()
        self._neg_prompt_embeds_mask = self._neg_prompt_embeds_mask.cpu()

        # Free text encoder
        if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
            pipe.text_encoder.to("cpu")
            del pipe.text_encoder
            pipe.text_encoder = None
        torch.cuda.empty_cache()
        import gc; gc.collect()
        self.emitter.info(f"  Text encoder freed: {torch.cuda.memory_allocated()/1e9:.1f}GB used")

        # --- 2. VAE-encode image ---
        pipe.vae.to(device)
        vae_input = img_tensor.unsqueeze(2).to(device=device, dtype=model_dtype)

        with torch.no_grad():
            image_latents = _retrieve_latents(pipe.vae.encode(vae_input), generator=generator)

        # Per-channel mean/std normalization
        latent_channels = pipe.vae.config.z_dim
        latents_mean = (
            torch.tensor(pipe.vae.config.latents_mean)
            .view(1, latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        latents_std = (
            torch.tensor(pipe.vae.config.latents_std)
            .view(1, latent_channels, 1, 1, 1)
            .to(image_latents.device, image_latents.dtype)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        # Pack 2x2: (B, C, 1, H, W) → (B, L, C*4)
        lat_h, lat_w = image_latents.shape[3], image_latents.shape[4]
        packed = pipe._pack_latents(image_latents, 1, latent_channels, lat_h, lat_w)
        self._image_latents_packed = packed.to(torch.float32).cpu()

        # Build img_shapes for transformer
        noise_ph = self._latent_height // 2
        noise_pw = self._latent_width // 2
        ref_ph = lat_h // 2
        ref_pw = lat_w // 2
        self._img_shapes = [[(1, noise_ph, noise_pw), (1, ref_ph, ref_pw)]]

        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return self._image_latents_packed.to(device)

    def mask_to_latent_space(self, mask, latent_shape):
        """Pixel mask → packed mask (1, L, 1) with 2x2 pooling."""
        mask_latent = torch.nn.functional.interpolate(
            mask, size=(self._latent_height, self._latent_width), mode="nearest",
        ).to(mask.device, torch.float32)

        _, _, h, w = mask_latent.shape
        mask_latent = mask_latent.view(1, 1, h // 2, 2, w // 2, 2)
        mask_latent = mask_latent.mean(dim=(1, 3, 5))
        mask_latent = (mask_latent > 0.5).float()
        return mask_latent.reshape(1, -1, 1)

    def prepare_timesteps(self, num_steps):
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
            calculate_shift, retrieve_timesteps,
        )
        device = self.device

        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        noise_seq_len = (self._latent_height // 2) * (self._latent_width // 2)
        mu = calculate_shift(
            noise_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, _ = retrieve_timesteps(
            self.pipe.scheduler, num_steps, device, sigmas=sigmas, mu=mu,
        )
        flow_ts = self.pipe.scheduler.sigmas.to(device)[:-1]
        timesteps = timesteps[:len(flow_ts)]
        return timesteps, flow_ts

    def predict_x0(self, x, flow_t, guidance_scale, cfg_big):
        device = self.device
        seq_len = x.shape[1]
        model_dtype = self.dtype

        # Concat reference image
        ref = self._image_latents_packed.to(device, x.dtype)
        latent_model_input = torch.cat([x, ref], dim=1)

        timestep = torch.full((x.shape[0],), flow_t, device=device, dtype=model_dtype)

        guidance = None
        if getattr(self.pipe.transformer.config, "guidance_embeds", False):
            guidance = torch.full([x.shape[0]], guidance_scale, device=device, dtype=torch.float32)

        with torch.no_grad():
            # Conditional
            with self.pipe.transformer.cache_context("cond"):
                v_cond = self.pipe.transformer(
                    hidden_states=latent_model_input.to(model_dtype),
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=self._prompt_embeds.to(device),
                    encoder_hidden_states_mask=self._prompt_embeds_mask.to(device),
                    img_shapes=self._img_shapes,
                    return_dict=False,
                )[0][:, :seq_len]

            # Unconditional
            with self.pipe.transformer.cache_context("uncond"):
                v_uncond = self.pipe.transformer(
                    hidden_states=latent_model_input.to(model_dtype),
                    timestep=timestep,
                    guidance=guidance,
                    encoder_hidden_states=self._neg_prompt_embeds.to(device),
                    encoder_hidden_states_mask=self._neg_prompt_embeds_mask.to(device),
                    img_shapes=self._img_shapes,
                    return_dict=False,
                )[0][:, :seq_len]

        # CFG with norm rescaling
        v_cond_f = v_cond.float()
        v_uncond_f = v_uncond.float()
        v_cfg = v_uncond_f + guidance_scale * (v_cond_f - v_uncond_f)
        v_big_raw = v_uncond_f + cfg_big * (v_cond_f - v_uncond_f)

        cond_norm = torch.norm(v_cond_f, dim=-1, keepdim=True)
        cfg_norm = torch.norm(v_cfg, dim=-1, keepdim=True).clamp(min=1e-8)
        big_norm = torch.norm(v_big_raw, dim=-1, keepdim=True).clamp(min=1e-8)

        v_cfg = v_cfg * (cond_norm / cfg_norm)
        v_big = v_big_raw * (cond_norm / big_norm)

        x0 = x.float() - flow_t * v_cfg
        x0_big = x.float() - flow_t * v_big
        return x0, x0_big

    def decode_latents(self, latents):
        device = self.device
        pipe = self.pipe
        pipe.vae.to(device)

        with torch.no_grad():
            unpacked = pipe._unpack_latents(
                latents.to(device), self._pixel_height, self._pixel_width, pipe.vae_scale_factor,
            )

            latent_channels = pipe.vae.config.z_dim
            latents_mean = (
                torch.tensor(pipe.vae.config.latents_mean)
                .view(1, latent_channels, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            latents_std_inv = (
                1.0 / torch.tensor(pipe.vae.config.latents_std)
                .view(1, latent_channels, 1, 1, 1)
                .to(unpacked.device, unpacked.dtype)
            )
            unpacked = unpacked / latents_std_inv + latents_mean

            img = pipe.vae.decode(unpacked.to(self.dtype), return_dict=False)[0][:, :, 0]
            pil = pipe.image_processor.postprocess(img, output_type="pil")

        pipe.vae.to("cpu")
        torch.cuda.empty_cache()
        return pil[0] if isinstance(pil, list) else pil
