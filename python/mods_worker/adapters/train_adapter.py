import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

from mods_worker.protocol import EventEmitter

_STEP_RE = re.compile(r"step\s*[:=]?\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE)
_LOSS_RE = re.compile(r"loss\s*[:=]?\s*([0-9eE+\-.]+)", re.IGNORECASE)


def _build_train_command(config_path: Path) -> List[str]:
    env_cmd = os.getenv("MODS_AITOOLKIT_TRAIN_CMD", "").strip()
    if env_cmd:
        env_cmd = env_cmd.replace("{config}", str(config_path)).replace("{python}", sys.executable)
        return shlex.split(env_cmd)

    return [sys.executable, "-m", "toolkit.job", "--config", str(config_path)]


def spec_to_aitoolkit_config(spec: dict) -> dict:
    """Translate a TrainJobSpec (parsed from YAML) into ai-toolkit's config format.

    This is the single place to maintain the mapping between mods spec fields
    and ai-toolkit's expected YAML configuration.
    """
    params = spec.get("params", {})
    dataset = spec.get("dataset", {})
    model = spec.get("model", {})
    output = spec.get("output", {})

    config = {
        "job": "extension",
        "config": {
            "name": output.get("lora_name", "lora-output"),
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": output.get("destination_dir", "output"),
                    "device": "cuda:0",
                    "trigger_word": params.get("trigger_word", "OHWX"),
                    "network": {
                        "type": "lora",
                        "linear": params.get("rank", 16),
                        "linear_alpha": params.get("rank", 16),
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": params.get("steps", 2000),
                        "max_step_saves_to_keep": 1,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset.get("path", ""),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "resolution": params.get("resolution", 1024),
                            "default_caption": params.get("trigger_word", "OHWX"),
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": params.get("steps", 2000),
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "flowmatch",
                        "optimizer": params.get("optimizer", "adamw8bit"),
                        "lr": params.get("learning_rate", 1e-4),
                    },
                    "model": {
                        "name_or_path": model.get("base_model_id", "flux-schnell"),
                        "quantize": params.get("quantize", True),
                    },
                    "sample": {
                        "sampler": "flowmatch",
                        "sample_every": params.get("steps", 2000),
                        "width": params.get("resolution", 1024),
                        "height": params.get("resolution", 1024),
                        "prompts": [],
                        "neg": "",
                        "seed": params.get("seed") or 42,
                        "walk_seed": True,
                        "guidance_scale": 4,
                        "sample_steps": 20,
                    },
                }
            ],
        },
    }

    if params.get("seed") is not None:
        config["config"]["process"][0]["train"]["seed"] = params["seed"]

    return config


def scan_output_artifacts(output_dir: str, emitter: EventEmitter) -> None:
    """After training, scan output directory for .safetensors files and emit artifact events."""
    import glob
    import hashlib

    pattern = os.path.join(output_dir, "**", "*.safetensors")
    for filepath in glob.glob(pattern, recursive=True):
        path = Path(filepath)
        size_bytes = path.stat().st_size

        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)

        emitter.artifact(
            path=str(path),
            sha256=sha256.hexdigest(),
            size_bytes=size_bytes,
        )


def run_train(config_path: Path, emitter: EventEmitter) -> int:
    if not config_path.exists():
        emitter.error(
            "SPEC_VALIDATION_FAILED",
            f"Training config not found: {config_path}",
            recoverable=False,
        )
        return 2

    # Try to load as a full TrainJobSpec first, fall back to direct config
    output_dir = None
    try:
        import yaml
        with open(config_path) as f:
            spec = yaml.safe_load(f)
        if isinstance(spec, dict) and "params" in spec:
            # This is a full TrainJobSpec — translate to ai-toolkit config
            aitk_config = spec_to_aitoolkit_config(spec)
            output_dir = spec.get("output", {}).get("destination_dir")
            # Write translated config to a temp file
            import tempfile
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
                yaml.dump(aitk_config, tmp)
                effective_config_path = Path(tmp.name)
        else:
            effective_config_path = config_path
    except ImportError:
        effective_config_path = config_path
    except Exception:
        effective_config_path = config_path

    cmd = _build_train_command(effective_config_path)
    emitter.job_started(config=str(config_path), command=cmd)

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as exc:
        emitter.error(
            "AITOOLKIT_EXEC_NOT_FOUND",
            f"Could not execute ai-toolkit command: {exc}",
            recoverable=False,
        )
        return 127
    except Exception as exc:
        emitter.error(
            "AITOOLKIT_EXEC_FAILED",
            str(exc),
            recoverable=False,
        )
        return 1

    last_step = None
    for raw_line in process.stdout or []:
        line = raw_line.strip()
        if not line:
            continue

        emitter.info(line)

        step_match = _STEP_RE.search(line)
        if step_match:
            step = int(step_match.group(1))
            total_steps = int(step_match.group(2))
            if last_step != step:
                loss = None
                loss_match = _LOSS_RE.search(line)
                if loss_match:
                    try:
                        loss = float(loss_match.group(1))
                    except ValueError:
                        pass
                emitter.progress(
                    stage="train",
                    step=step,
                    total_steps=total_steps,
                    loss=loss,
                )
                last_step = step

    code = process.wait()
    if code == 0:
        # Scan for output artifacts
        if output_dir and os.path.isdir(output_dir):
            scan_output_artifacts(output_dir, emitter)
        emitter.completed("ai-toolkit training command finished")
    else:
        emitter.error(
            "TRAINING_FAILED",
            f"ai-toolkit process exited with code {code}",
            recoverable=False,
            details={"exit_code": code},
        )
    return code
