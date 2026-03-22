"""Mask-blend inpainting via monkey-patched prepare_latents.

Approach D: don't fight the pipeline. Instead:
1. Pre-encode prompt with the ORIGINAL image (CLIP style anchor)
2. VAE-encode the padded image, create latent mask, seed with noise
3. Monkey-patch prepare_latents to inject our seeded latents
4. Use callback_on_step_end for per-step mask re-blending
5. Call pipe() normally — it handles dtype, hooks, VRAM, packing

All blending happens at full VAE resolution (B, C_vae, H, W).
The callback unpacks sequence → VAE spatial → blend → repack.
Image ID channels in the sequence tensor are never touched.
"""

import torch
import numpy as np
from PIL import Image
from diffusers.image_processor import VaeImageProcessor


# ---------------------------------------------------------------------------
# Symmetric VAE ↔ transformer format conversions
# ---------------------------------------------------------------------------

def vae_to_transformer(latents, c_vae):
    """(B, C_vae, H, W) → (B, C_vae*4, H/2, W/2)  — 2x2 spatial fold."""
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)  # (B, C, 2, 2, H/2, W/2)
    return latents.reshape(b, c * 4, h // 2, w // 2)


def transformer_to_vae(latents, c_vae):
    """(B, C_vae*4, H/2, W/2) → (B, C_vae, H, W)  — exact inverse."""
    b, c4, hh, hw = latents.shape
    latents = latents.view(b, c_vae, 2, 2, hh, hw)
    latents = latents.permute(0, 1, 4, 2, 5, 3)  # (B, C, H/2, 2, W/2, 2)
    return latents.reshape(b, c_vae, hh * 2, hw * 2)


def spatial_to_seq(latents):
    """(B, C, H, W) → (B, H*W, C)"""
    b, c, h, w = latents.shape
    return latents.reshape(b, c, h * w).permute(0, 2, 1)


def seq_to_spatial(latents, h, w):
    """(B, H*W, C) → (B, C, H, W)"""
    return latents.permute(0, 2, 1).reshape(latents.shape[0], -1, h, w)


# ---------------------------------------------------------------------------
# VAE scaling — must be exact inverse of decode: (lat / scale) + shift
# ---------------------------------------------------------------------------

def _encode_and_scale(pipe, image_tensor, generator):
    """VAE encode + scale using the pipeline's own method.

    Different pipelines scale differently:
    - Klein: patchify → BN normalize (no scaling_factor/shift_factor)
    - Flux/Z-Image: (raw - shift_factor) * scaling_factor
    - SDXL: raw * scaling_factor

    Uses the pipeline's _encode_vae_image if available to match exactly.
    """
    # Use pipeline's own encode method if available (Klein, Flux2)
    if hasattr(pipe, '_encode_vae_image'):
        with torch.no_grad():
            return pipe._encode_vae_image(image=image_tensor, generator=generator)

    # Fallback: manual encode + standard scaling
    with torch.no_grad():
        enc = pipe.vae.encode(image_tensor)
        raw = enc.latent_dist.mode() if hasattr(enc, 'latent_dist') else enc.mode()

    lat = raw.clone()
    if hasattr(pipe.vae.config, 'shift_factor') and pipe.vae.config.shift_factor:
        lat = lat - pipe.vae.config.shift_factor
    if hasattr(pipe.vae.config, 'scaling_factor') and pipe.vae.config.scaling_factor:
        lat = lat * pipe.vae.config.scaling_factor
    return lat


def _is_flow_matching(scheduler) -> bool:
    cls_name = type(scheduler).__name__
    return ("FlowMatch" in cls_name or "Flow" in cls_name or
            getattr(getattr(scheduler, "config", None), "prediction_type", None) == "flow_matching")


def _detect_packing(pipe):
    """Returns 'flux', 'klein', or 'none'."""
    if not hasattr(pipe, '_pack_latents'):
        return "none"
    import inspect
    src = inspect.getsource(pipe._pack_latents)
    if "// 2" in src and "permute" in src:
        return "flux"
    return "klein"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_mask_blend(
    pipe,
    prompt: str,
    init_image: Image.Image,
    mask_image: Image.Image,
    condition_image: Image.Image | None,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int | None,
    arch: str,
    emitter,
    negative_prompt: str = "",
    gen_kwargs: dict | None = None,
) -> Image.Image:
    if gen_kwargs is None:
        gen_kwargs = {}

    device = getattr(pipe, '_execution_device', None) or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = pipe.vae.dtype
    vae_scale = getattr(pipe, 'vae_scale_factor', 8)
    packing = _detect_packing(pipe)

    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(seed)

    # --- 1. CLIP-encode prompt with ORIGINAL image ---
    encoder_image = condition_image or init_image
    emitter.info("Encoding prompt with original image...")
    prompt_embeds, neg_embeds = _encode_prompt_split(
        pipe, prompt, negative_prompt, encoder_image, arch, device,
    )

    # --- 2. VAE-encode + seeded latents ---
    emitter.info("Preparing latent canvas...")

    init_resized = init_image.convert("RGB").resize((width, height), Image.LANCZOS)
    mask_resized = mask_image.convert("L").resize((width, height), Image.NEAREST)
    mask_np = np.array(mask_resized).astype(np.float32) / 255.0

    # Pixel noise in masked areas
    init_np = np.array(init_resized).astype(np.float32) / 255.0
    rng = np.random.RandomState(seed % (2**31) if seed else 42)
    pixel_noise = rng.rand(*init_np.shape).astype(np.float32)
    mask_3ch = mask_np[:, :, np.newaxis]
    padded_np = (1.0 - mask_3ch) * init_np + mask_3ch * pixel_noise
    padded_pil = Image.fromarray((np.clip(padded_np, 0, 1) * 255).astype(np.uint8))

    img_processor = VaeImageProcessor(vae_scale_factor=vae_scale)
    padded_tensor = img_processor.preprocess(padded_pil, height=height, width=width).to(device=device, dtype=dtype)
    clean_tensor = img_processor.preprocess(init_resized, height=height, width=width).to(device=device, dtype=dtype)

    padded_latents = _encode_and_scale(pipe, padded_tensor, generator)
    clean_latents = _encode_and_scale(pipe, clean_tensor, generator)

    # The encode may return raw VAE output (B, 16, 64, 64) or already-folded
    # pipeline format (B, 128, 32, 32) depending on whether _encode_vae_image
    # was used. Detect by checking if the spatial dims are half of expected.
    c_enc = clean_latents.shape[1]
    h_enc = clean_latents.shape[2]
    w_enc = clean_latents.shape[3]
    expected_h = height // vae_scale  # e.g. 64 for 512px
    already_folded = (h_enc < expected_h)  # 32 < 64 → already folded

    if already_folded:
        # Pipeline's _encode_vae_image already patchified/BN-normalized
        # Latents are in pipeline's internal spatial format
        c_vae = c_enc // 4   # 128 // 4 = 32 (Klein's effective VAE channels)
        latent_h = h_enc      # folded spatial
        latent_w = w_enc
        c_folded = c_enc      # already folded channel count
        hh, hw = h_enc, w_enc
        emitter.info(f"Encoded latents (pre-folded): ({c_enc}, {h_enc}, {w_enc}), "
                     f"mean={clean_latents.mean():.4f} std={clean_latents.std():.4f}")
    else:
        # Raw VAE output — need to fold for sequence-packing pipelines
        c_vae = c_enc          # 16
        latent_h = h_enc       # full spatial
        latent_w = w_enc
        c_folded = c_vae * 4   # 64 after fold
        hh = h_enc // 2
        hw = w_enc // 2
        emitter.info(f"Encoded latents (raw): ({c_enc}, {h_enc}, {w_enc}), "
                     f"mean={clean_latents.mean():.4f} std={clean_latents.std():.4f}")

    # Mask at the encoded spatial resolution
    mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
    latent_mask = torch.nn.functional.interpolate(
        mask_tensor, size=(h_enc, w_enc), mode="bilinear", align_corners=False,
    ).to(device=device, dtype=clean_latents.dtype)

    # Noise at encoded resolution
    noise = torch.randn(
        clean_latents.shape, generator=generator, device="cpu", dtype=torch.float32,
    ).to(device=device, dtype=clean_latents.dtype)

    # Seeded: preserved = encoded original, masked = noise
    seeded_latents = latent_mask * noise + (1.0 - latent_mask) * padded_latents

    emitter.info(f"Packing: {packing}, already_folded: {already_folded}, mask coverage: {mask_np.mean():.0%}")

    # --- 3. Monkey-patch prepare_latents ---
    real_prepare = pipe.prepare_latents

    def patched_prepare_latents(*args, **kwargs):
        result = real_prepare(*args, **kwargs)
        packed = result[0] if isinstance(result, tuple) else result

        if packed.dim() == 3:
            # Sequence format: get to spatial → seq → inject
            if already_folded:
                sl_spatial = seeded_latents  # already in (B, C_folded, H, W)
            else:
                sl_spatial = vae_to_transformer(seeded_latents, c_vae)
            sl_seq = spatial_to_seq(sl_spatial)
            sl_seq = sl_seq.to(device=packed.device, dtype=packed.dtype)
            packed[:, :, :sl_seq.shape[-1]] = sl_seq
        else:
            packed.copy_(seeded_latents.to(device=packed.device, dtype=packed.dtype))

        return result

    pipe.prepare_latents = patched_prepare_latents

    # --- 4. Callback: blend at full VAE resolution ---
    is_flow = _is_flow_matching(pipe.scheduler)

    def reblend_callback(pipe_ref, step_index, timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        lat_dev = latents.device
        lat_dt = latents.dtype

        cl = clean_latents.to(device=lat_dev, dtype=lat_dt)
        ns = noise.to(device=lat_dev, dtype=lat_dt)
        mk = latent_mask.to(device=lat_dev, dtype=lat_dt).expand_as(cl)
        noised_orig = _renoise(cl, ns, pipe_ref.scheduler, step_index, timestep, is_flow, lat_dev, lat_dt)

        if latents.dim() == 3:
            # Sequence format: extract our channels → unpack → blend → repack
            b, seq_len, c_total = latents.shape

            vae_seq = latents[:, :, :c_folded]
            vae_spatial = seq_to_spatial(vae_seq, hh, hw)  # (B, c_folded, hh, hw)

            if already_folded:
                # Blend directly at folded resolution (clean/noise/mask match)
                blended = mk * vae_spatial + (1.0 - mk) * noised_orig
                blended_seq = spatial_to_seq(blended).to(lat_dt)
            else:
                # Unfold → blend at full VAE res → refold
                vae_full = transformer_to_vae(vae_spatial, c_vae)
                blended = mk * vae_full + (1.0 - mk) * noised_orig
                blended_folded = vae_to_transformer(blended, c_vae)
                blended_seq = spatial_to_seq(blended_folded).to(lat_dt)

            latents[:, :, :c_folded] = blended_seq
            callback_kwargs["latents"] = latents
        else:
            # 4D spatial: blend directly
            callback_kwargs["latents"] = mk * latents + (1.0 - mk) * noised_orig

        return callback_kwargs

    # --- 5. Pipeline kwargs ---
    pipe_kwargs = dict(gen_kwargs)
    pipe_kwargs["prompt"] = prompt
    pipe_kwargs["num_inference_steps"] = steps
    pipe_kwargs["generator"] = generator
    pipe_kwargs["width"] = width
    pipe_kwargs["height"] = height
    pipe_kwargs["callback_on_step_end"] = reblend_callback
    pipe_kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]

    if prompt_embeds is not None and arch in ("qwen_image", "qwen_image_edit"):
        pipe_kwargs["prompt_embeds"] = prompt_embeds
        pipe_kwargs.pop("prompt", None)
        if neg_embeds is not None:
            pipe_kwargs["negative_prompt_embeds"] = neg_embeds

    _add_arch_kwargs(pipe_kwargs, arch, guidance)

    # --- 6. Run ---
    emitter.info(f"Denoising {steps} steps...")
    try:
        result = pipe(**pipe_kwargs)
        image = result.images[0]
    finally:
        pipe.prepare_latents = real_prepare
    return image


def _renoise(clean, noise, scheduler, step_index, timestep, is_flow, device, dtype):
    """Re-noise clean latents to the current noise level."""
    if is_flow:
        sigmas = scheduler.sigmas
        next_idx = step_index + 1
        if next_idx < len(sigmas):
            sigma = sigmas[next_idx].to(device=device, dtype=dtype)
        else:
            sigma = torch.tensor(0.0, device=device, dtype=dtype)
        sigma = sigma.reshape(1, 1, 1, 1)
        return sigma * noise + (1.0 - sigma) * clean
    else:
        return scheduler.add_noise(clean, noise, timestep.unsqueeze(0))


def _add_arch_kwargs(kwargs, arch, guidance):
    if arch in ("qwen_image", "qwen_image_edit"):
        kwargs["true_cfg_scale"] = guidance
        kwargs.setdefault("negative_prompt", " ")
    elif arch == "chroma":
        kwargs["guidance_scale"] = guidance
        kwargs.setdefault("negative_prompt", "low quality, ugly, deformed")
    else:
        kwargs["guidance_scale"] = guidance


def _encode_prompt_split(pipe, prompt, negative_prompt, original_image, arch, device):
    """Encode prompt with original image for CLIP style anchoring."""
    import inspect

    call_sig = inspect.signature(pipe.__call__)
    if "prompt_embeds" not in call_sig.parameters:
        return None, None

    encode_sig = inspect.signature(pipe.encode_prompt)
    encode_params = set(encode_sig.parameters.keys())

    try:
        if "image" in encode_params:
            result = pipe.encode_prompt(prompt=prompt, image=original_image, device=device)
        else:
            kw = {"prompt": prompt, "device": device}
            if "prompt_2" in encode_params:
                kw["prompt_2"] = prompt
            result = pipe.encode_prompt(**kw)

        prompt_embeds = result[0] if isinstance(result, tuple) else result

        neg_embeds = None
        if negative_prompt:
            if "image" in encode_params:
                neg_result = pipe.encode_prompt(prompt=negative_prompt, image=original_image, device=device)
            else:
                nkw = {"prompt": negative_prompt, "device": device}
                if "prompt_2" in encode_params:
                    nkw["prompt_2"] = negative_prompt
                neg_result = pipe.encode_prompt(**nkw)
            neg_embeds = neg_result[0] if isinstance(neg_result, tuple) else neg_result

        return prompt_embeds, neg_embeds
    except Exception:
        import traceback
        traceback.print_exc()
        return None, None
