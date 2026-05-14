#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from tools.inference.test_q1_student_video import main as backend_main


REPO_ROOT = Path(__file__).resolve().parent


def default_generation_ckpt() -> Path:
    return (
        REPO_ROOT
        / "omni_ckpts"
        / "hf_mobile_ov"
        / "stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt"
    )


def resolve_path(raw: str) -> Path:
    return Path(raw).expanduser().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mobile-OV generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Mobile-OV checkpoint (.pt). Defaults to MOBILEOV_GENERATION_CKPT or the local 60k checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("MOBILEOV_SANA_CHECKPOINT_DIR", "omni_ckpts/sana_video_2b_480p"),
        help="Local SANA checkpoint directory.",
    )
    parser.add_argument(
        "--smolvlm2-ckpt-path",
        type=str,
        default=os.environ.get("SMOLVLM2_CKPT_PATH", "omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"),
        help="Local SmolVLM2 checkpoint used by the bridge.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of video frames to generate.")
    parser.add_argument("--height", type=int, default=480, help="Output height.")
    parser.add_argument("--width", type=int, default=832, help="Output width.")
    parser.add_argument("--steps", type=int, default=24, help="Sampling steps.")
    parser.add_argument("--cfg-scale", type=float, default=6.0, help="CFG scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Inference dtype.")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt.")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    args = parse_args()

    ckpt_raw = args.checkpoint or os.environ.get("MOBILEOV_GENERATION_CKPT")
    checkpoint = resolve_path(ckpt_raw) if ckpt_raw else default_generation_ckpt().resolve()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Generation checkpoint not found: {checkpoint}")

    backend_argv = [
        "--bridge-ckpt",
        str(checkpoint),
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--smolvlm2-ckpt-path",
        args.smolvlm2_ckpt_path,
        "--prompt",
        args.prompt,
        "--output-dir",
        str(output_dir),
        "--num-frames",
        str(args.num_frames),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--steps",
        str(args.steps),
        "--cfg-scale",
        str(args.cfg_scale),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--negative-prompt",
        args.negative_prompt,
    ]

    result = backend_main(backend_argv)
    return int(0 if result is None else result)


if __name__ == "__main__":
    raise SystemExit(main())
