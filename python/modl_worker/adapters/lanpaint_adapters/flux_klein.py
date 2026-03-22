"""Flux2 Klein adapter for LanPaint.

Key conventions:
- Packed latent sequences (B, L, C) — not spatial
- Edit-style: reference image concatenated at every forward pass
- Cache-context based CFG (separate cond/uncond passes)
- Standard flow matching: x0 = x - flow_t * v
- Decode: unpack → BN denorm → unpatchify → VAE

Reference: LanPaint-diffusers FluxKleinAdapter (charrywhite/LanPaint-diffusers).
"""

from typing import Tuple

import numpy as np
import torch
from PIL import Image

from .base import ModelAdapter


class FluxKleinAdapter(ModelAdapter):
    """LanPaint adapter for Flux2 Klein (4B and 9B)."""

    def __init__(self, pipe, emitter):
        super().__init__(pipe, emitter)
        self._prompt_embeds = None
        self._text_ids = None
        self._neg_prompt_embeds = None
        self._neg_text_ids = None
        self._y_packed = None  # reference image latent (packed)
        self._latent_ids = None
        self._ref_image_ids = None
        self._height = None
        self._width = None

    @property
    def distilled(self):
        return True  # Klein is distilled

    def encode_prompt(self, prompt, negative_prompt):
        device = self.device
        self.pipe.text_encoder.to(device)

        with torch.no_grad():
            self._prompt_embeds, self._text_ids = self.pipe.encode_prompt(
                prompt=prompt, device=device,
            )
            self._neg_prompt_embeds, self._neg_text_ids = self.pipe.encode_prompt(
                prompt=negative_prompt or "", device=device,
            )

        # Move to CPU
        self._prompt_embeds = self._prompt_embeds.cpu()
        self._text_ids = self._text_ids.cpu()
        self._neg_prompt_embeds = self._neg_prompt_embeds.cpu()
        self._neg_text_ids = self._neg_text_ids.cpu()

        self.pipe.text_encoder.to("cpu")
        if hasattr(self.pipe, 'tokenizer') and self.pipe.tokenizer is not None:
            pass  # tokenizer is CPU-only
        from modl_worker.device import empty_cache
        empty_cache()
        import gc; gc.collect()

    def encode_image(self, img_tensor, generator):
        device = self.device
        self.pipe.vae.to(device)

        with torch.no_grad():
            cpu_gen = torch.Generator("cpu").manual_seed(generator.initial_seed())
            image_latents, ref_image_ids = self.pipe.prepare_image_latents(
                images=[img_tensor.to(device)],
                batch_size=1,
                generator=cpu_gen,
                device=device,
                dtype=self.pipe.vae.dtype,
            )
        self._y_packed = image_latents.to(torch.float32).cpu()
        self._ref_image_ids = ref_image_ids.cpu()

        # Prepare noise latent shape + IDs
        num_ch = self.pipe.transformer.config.in_channels // 4
        _, latent_ids = self.pipe.prepare_latents(
            batch_size=1,
            num_latents_channels=num_ch,
            height=self._height or img_tensor.shape[-2],
            width=self._width or img_tensor.shape[-1],
            dtype=torch.float32,
            device=device,
            generator=generator,
            latents=None,
        )
        self._latent_ids = latent_ids.cpu()

        self.pipe.vae.to("cpu")
        from modl_worker.device import empty_cache
        empty_cache()
        return self._y_packed.to(device)

    def mask_to_latent_space(self, mask, latent_shape):
        """Pixel mask → packed latent mask (1, L, 1)."""
        device = self.device
        # Unpack to get spatial dims
        y_unpacked = self.pipe._unpack_latents_with_ids(
            self._y_packed.to(device), self._ref_image_ids.to(device),
        )
        _, _, ph, pw = y_unpacked.shape
        mask_latent = torch.nn.functional.interpolate(
            mask, size=(ph, pw), mode="nearest",
        ).to(device, torch.float32).reshape(1, -1, 1)
        return mask_latent

    def prepare_timesteps(self, num_steps):
        from diffusers.pipelines.flux2.pipeline_flux2_klein import (
            compute_empirical_mu, retrieve_timesteps,
        )
        device = self.device

        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        image_seq_len = self._y_packed.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_steps)
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
        y_gpu = self._y_packed.to(device, x.dtype)
        latent_model_input = torch.cat([x, y_gpu], dim=1)
        img_ids = torch.cat(
            [self._latent_ids.to(device), self._ref_image_ids.to(device)], dim=1
        )
        t_tensor = torch.full((x.shape[0],), flow_t, device=device, dtype=model_dtype)

        # Conditional pass
        with torch.no_grad():
            with self.pipe.transformer.cache_context("cond"):
                v_cond = self.pipe.transformer(
                    hidden_states=latent_model_input.to(model_dtype),
                    timestep=t_tensor,
                    guidance=None,
                    encoder_hidden_states=self._prompt_embeds.to(device),
                    txt_ids=self._text_ids.to(device),
                    img_ids=img_ids,
                    return_dict=False,
                )[0][:, :seq_len]

            # Unconditional pass
            with self.pipe.transformer.cache_context("uncond"):
                v_uncond = self.pipe.transformer(
                    hidden_states=latent_model_input.to(model_dtype),
                    timestep=t_tensor,
                    guidance=None,
                    encoder_hidden_states=self._neg_prompt_embeds.to(device),
                    txt_ids=self._neg_text_ids.to(device),
                    img_ids=img_ids,
                    return_dict=False,
                )[0][:, :seq_len]

        # Dual CFG + x0 (standard flow: x0 = x - t*v)
        v_cfg = v_uncond.float() + guidance_scale * (v_cond.float() - v_uncond.float())
        v_big = v_uncond.float() + cfg_big * (v_cond.float() - v_uncond.float())

        x0 = x.float() - flow_t * v_cfg
        x0_big = x.float() - flow_t * v_big
        return x0, x0_big

    def decode_latents(self, latents):
        device = self.device
        self.pipe.vae.to(device)
        model_dtype = self.dtype

        with torch.no_grad():
            unpacked = self.pipe._unpack_latents_with_ids(
                latents.to(device), self._latent_ids.to(device)
            )

            # BN denorm
            bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(
                unpacked.device, unpacked.dtype,
            )
            bn_std = torch.sqrt(
                self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
            ).to(unpacked.device, unpacked.dtype)
            unpacked = unpacked * bn_std + bn_mean

            spatial = self.pipe._unpatchify_latents(unpacked)
            img = self.pipe.vae.decode(spatial.to(model_dtype), return_dict=False)[0]
            pil = self.pipe.image_processor.postprocess(img, output_type="pil")

        self.pipe.vae.to("cpu")
        from modl_worker.device import empty_cache
        empty_cache()
        return pil[0] if isinstance(pil, list) else pil
