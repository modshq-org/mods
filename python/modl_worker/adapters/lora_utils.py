"""Shared LoRA key conversion utilities.

Handles the diffusion_model.* -> transformer.* key prefix conversion
required when loading ai-toolkit or some CivitAI LoRAs with diffusers.
"""

from __future__ import annotations

import os
import tempfile
import logging

logger = logging.getLogger(__name__)


def convert_lora_keys_if_needed(lora_path: str) -> tuple[str | None, str | None]:
    """Convert LoRA state dict keys from diffusion_model.* to transformer.*.

    If the LoRA has keys with the ``diffusion_model.`` prefix (ai-toolkit
    format), converts them to ``transformer.`` (diffusers format), saves to
    a temp file, and returns (tmp_path, None).

    If no conversion is needed, returns (None, None).

    On failure, returns (None, error_message) so callers can warn and
    continue without LoRA.

    The caller is responsible for deleting the temp file after use.
    """
    try:
        from safetensors.torch import load_file, save_file

        raw_sd = load_file(lora_path)

        old_prefix = "diffusion_model."
        new_prefix = "transformer."
        needs_conversion = any(k.startswith(old_prefix) for k in raw_sd)

        if not needs_conversion:
            return None, None

        converted = {
            new_prefix + k[len(old_prefix):] if k.startswith(old_prefix) else k: v
            for k, v in raw_sd.items()
        }

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
            tmp_path = tmp.name
            save_file(converted, tmp_path)

        return tmp_path, None

    except Exception as exc:
        return None, str(exc)


def load_lora_with_conversion(
    pipeline,
    lora_path: str,
    lora_weight: float = 1.0,
    emitter=None,
) -> bool:
    """Load a LoRA onto a pipeline, with automatic key conversion fallback.

    Tries direct loading first.  On failure, attempts key prefix conversion
    (diffusion_model.* -> transformer.*).  On second failure, logs a warning
    and returns False so the caller can continue without LoRA.

    Returns True if the LoRA was successfully loaded and fused, False otherwise.
    """
    lora_dir = os.path.dirname(lora_path)
    lora_file = os.path.basename(lora_path)

    try:
        pipeline.load_lora_weights(lora_dir, weight_name=lora_file)
        pipeline.fuse_lora(lora_scale=lora_weight)
        return True
    except Exception as first_err:
        if emitter:
            emitter.info(f"  Retrying with key conversion (first error: {first_err})")

        tmp_path, convert_err = convert_lora_keys_if_needed(lora_path)

        if convert_err:
            _warn_lora_failed(emitter, convert_err)
            return False

        if tmp_path is None:
            # No conversion needed but direct load failed — incompatible LoRA
            _warn_lora_failed(emitter, str(first_err))
            return False

        try:
            pipeline.load_lora_weights(
                os.path.dirname(tmp_path),
                weight_name=os.path.basename(tmp_path),
            )
            pipeline.fuse_lora(lora_scale=lora_weight)
            return True
        except Exception as second_err:
            _warn_lora_failed(emitter, str(second_err))
            return False
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _warn_lora_failed(emitter, message: str) -> None:
    """Emit a warning about LoRA loading failure."""
    if emitter:
        emitter.warning(
            "LORA_INCOMPATIBLE",
            f"Could not load LoRA (incompatible with model?): {message}. "
            f"Generating without LoRA.",
        )
    else:
        logger.warning("LoRA load failed: %s — generating without LoRA", message)
