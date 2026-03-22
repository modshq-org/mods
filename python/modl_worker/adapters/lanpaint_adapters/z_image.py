"""Z-Image adapter for LanPaint.

Key convention: diffusers' Z-Image output is NEGATED vs ComfyUI.
ZImagePipeline line 265: noise_pred = -noise_pred.
Therefore: x0 = x + raw * sigma (not x - raw * sigma).

Transformer expects: list of (C, F, H, W), timestep = 1 - flow_t, list of embeds.
"""

from typing import Tuple

import torch
from PIL import Image

from .base import ModelAdapter


class ZImageAdapter(ModelAdapter):
    """LanPaint adapter for Z-Image (base and turbo)."""

    def __init__(self, pipe, emitter, is_turbo=False):
        super().__init__(pipe, emitter)
        self._is_turbo = is_turbo
        self._prompt_embeds = None
        self._neg_prompt_embeds = None

    @property
    def distilled(self):
        return self._is_turbo

    def encode_prompt(self, prompt, negative_prompt):
        device = self.device
        self.pipe.text_encoder.to(device)
        with torch.no_grad():
            pe, npe = self.pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=True,
                device=device,
            )
        # Move to CPU (tiny tensors)
        self._prompt_embeds = [p.cpu() for p in pe]
        self._neg_prompt_embeds = [n.cpu() for n in npe] if npe else None

        # Free text encoder completely
        self.pipe.text_encoder.to("cpu")
        del self.pipe.text_encoder
        self.pipe.text_encoder = None
        from modl_worker.device import empty_cache
        empty_cache()
        import gc; gc.collect()

    def encode_image(self, img_tensor, generator):
        device = self.device
        self.pipe.vae.to(device)
        img_tensor = img_tensor.to(device=device, dtype=self.pipe.vae.dtype)

        with torch.no_grad():
            dist = self.pipe.vae.encode(img_tensor)
            latent = dist.latent_dist.mode() if hasattr(dist, "latent_dist") else dist.mode()

        # Scale only (no shift) — matches ComfyUI's process_latent_in
        if hasattr(self.pipe.vae.config, "scaling_factor"):
            latent = latent * self.pipe.vae.config.scaling_factor

        self.pipe.vae.to("cpu")
        from modl_worker.device import empty_cache
        empty_cache()
        return latent.to(dtype=torch.float32)

    def prepare_timesteps(self, num_steps):
        from diffusers.pipelines.z_image.pipeline_z_image import calculate_shift, retrieve_timesteps

        device = self.device
        # Use shift=3.0 matching ComfyUI (diffusers ships 6.0)
        self.pipe.scheduler.config["shift"] = 3.0

        # Calculate sequence-length-dependent mu
        # Assume standard latent dims from the last encoded image
        vae_scale = self.pipe.vae_scale_factor * 2
        image_seq_len = 48 * 64 // 4  # default, overridden by actual dims

        mu = calculate_shift(
            image_seq_len,
            self.pipe.scheduler.config.get("base_image_seq_len", 256),
            self.pipe.scheduler.config.get("max_image_seq_len", 4096),
            self.pipe.scheduler.config.get("base_shift", 0.5),
            self.pipe.scheduler.config.get("max_shift", 1.15),
        )
        self.pipe.scheduler.sigma_min = 0.0
        timesteps, _ = retrieve_timesteps(self.pipe.scheduler, num_steps, device, mu=mu)
        flow_ts = self.pipe.scheduler.sigmas[:-1].to(device=device, dtype=torch.float32)
        return timesteps, flow_ts

    def predict_x0(self, x, flow_t, guidance_scale, cfg_big):
        device = self.device
        sigma = flow_t
        model_t = 1.0 - sigma  # Z-Image timestep: 0=noisy, 1=clean
        batch = x.shape[0]

        sigma_e = torch.tensor(sigma, device=device)
        while sigma_e.dim() < x.dim():
            sigma_e = sigma_e.unsqueeze(-1)

        timestep = torch.full((batch,), model_t, device=device, dtype=self.dtype)
        embeds_gpu = [e.to(device) for e in self._prompt_embeds]

        # Prepare input: add frame dim, unbind to list
        x_in = x.to(self.dtype)
        if x_in.dim() == 4:
            x_in = x_in.unsqueeze(2)
        x_list = list(x_in.unbind(dim=0))

        # Conditional forward
        with torch.no_grad():
            out_cond = self.pipe.transformer(x_list, timestep, embeds_gpu, return_dict=False)[0]
        raw_cond = torch.stack([t.float() for t in out_cond]).squeeze(2)

        # x0 = x + raw * sigma (diffusers negated convention)
        x0_cond = x.float() + raw_cond * sigma_e

        if guidance_scale <= 1.0 or self._neg_prompt_embeds is None:
            return x0_cond, x0_cond

        # Unconditional forward
        neg_gpu = [e.to(device) for e in self._neg_prompt_embeds]
        with torch.no_grad():
            out_uncond = self.pipe.transformer(x_list, timestep, neg_gpu, return_dict=False)[0]
        raw_uncond = torch.stack([t.float() for t in out_uncond]).squeeze(2)
        x0_uncond = x.float() + raw_uncond * sigma_e

        # Dual CFG
        x0_std = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
        x0_big = x0_uncond + cfg_big * (x0_cond - x0_uncond)
        return x0_std, x0_big

    def decode_latents(self, latents):
        device = self.device
        self.pipe.vae.to(device)
        with torch.no_grad():
            decoded = latents.to(self.pipe.vae.dtype)
            if hasattr(self.pipe.vae.config, "shift_factor"):
                decoded = decoded + self.pipe.vae.config.shift_factor
            if hasattr(self.pipe.vae.config, "scaling_factor"):
                decoded = decoded / self.pipe.vae.config.scaling_factor
            image_tensor = self.pipe.vae.decode(decoded, return_dict=False)[0]
        result = self.pipe.image_processor.postprocess(image_tensor, output_type="pil")[0]
        self.pipe.vae.to("cpu")
        from modl_worker.device import empty_cache
        empty_cache()
        return result
