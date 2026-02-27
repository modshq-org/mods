import argparse
import os
import sys
from pathlib import Path

from mods_worker.adapters import run_train, run_generate, run_caption
from mods_worker.protocol import EventEmitter, fatal


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mods_worker")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Run training adapter")
    train.add_argument("--config", required=True, help="Path to training config/spec yaml")
    train.add_argument("--job-id", default="", help="Job ID for event envelope")

    gen = sub.add_parser("generate", help="Run inference/generation adapter")
    gen.add_argument("--config", required=True, help="Path to generate spec yaml")
    gen.add_argument("--job-id", default="", help="Job ID for event envelope")

    cap = sub.add_parser("caption", help="Run auto-captioning adapter")
    cap.add_argument("--config", required=True, help="Path to caption spec yaml")
    cap.add_argument("--job-id", default="", help="Job ID for event envelope")

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    job_id = getattr(args, "job_id", "") or ""
    emitter = EventEmitter(source="mods_worker", job_id=job_id)

    if args.command == "train":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_train(config_path, emitter)

    if args.command == "generate":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_generate(config_path, emitter)

    if args.command == "caption":
        config_path = Path(args.config)
        emitter.job_accepted(worker_pid=os.getpid())
        return run_caption(config_path, emitter)

    fatal(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
