#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch


BUNDLE_FORMAT = "mobile_ov_bundle_v1"
INFERENCE_KEYS = (
    "step",
    "micro_step",
    "student_state",
    "dit_trainable_state",
    "dit_train_modules",
    "infer_hints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack Mobile-OV generation + SmolVLM2 into one inference weight.")
    parser.add_argument("--mobile-ov-checkpoint", required=True, help="Mobile-OV training/inference checkpoint.")
    parser.add_argument("--smolvlm2-checkpoint", required=True, help="Converted SmolVLM2 checkpoint.")
    parser.add_argument("--output", required=True, help="Output bundle .pt path.")
    parser.add_argument(
        "--keep-training-state",
        action="store_true",
        help="Keep optimizer/scheduler and any extra keys from the Mobile-OV checkpoint.",
    )
    return parser.parse_args()


def slim_mobile_ov_checkpoint(checkpoint: dict, *, keep_training_state: bool) -> dict:
    if keep_training_state:
        return checkpoint
    return {key: checkpoint[key] for key in INFERENCE_KEYS if key in checkpoint}


def main() -> int:
    args = parse_args()
    mobile_ov_path = Path(args.mobile_ov_checkpoint).expanduser().resolve()
    smolvlm2_path = Path(args.smolvlm2_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not mobile_ov_path.is_file():
        raise FileNotFoundError(f"Mobile-OV checkpoint not found: {mobile_ov_path}")
    if not smolvlm2_path.is_file():
        raise FileNotFoundError(f"SmolVLM2 checkpoint not found: {smolvlm2_path}")

    mobile_ov_checkpoint = torch.load(str(mobile_ov_path), map_location="cpu")
    if not isinstance(mobile_ov_checkpoint, dict):
        raise RuntimeError("Expected Mobile-OV checkpoint to be a dict.")

    bundle = {
        "format": BUNDLE_FORMAT,
        "mobile_ov_checkpoint": slim_mobile_ov_checkpoint(
            mobile_ov_checkpoint,
            keep_training_state=bool(args.keep_training_state),
        ),
        "smolvlm2_checkpoint_bytes": smolvlm2_path.read_bytes(),
        "meta": {
            "mobile_ov_checkpoint": str(mobile_ov_path),
            "smolvlm2_checkpoint": str(smolvlm2_path),
            "kept_training_state": bool(args.keep_training_state),
            "note": "SANA-video backbone is intentionally not bundled; it remains a public runtime dependency.",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, str(output_path))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
