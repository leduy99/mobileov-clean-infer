#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from nets.mobile_ov import (
    MobileOVModel,
    default_generation_ckpt,
    default_smolvlm2_ckpt,
    default_video_backbone_checkpoint_dir,
    resolve_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mobile-OV generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.environ.get("MOBILEOV_GENERATION_CKPT", str(default_generation_ckpt())),
        help="Mobile-OV checkpoint (.pt).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("VIDEO_BACKBONE_CHECKPOINT_DIR", str(default_video_backbone_checkpoint_dir())),
        help="Local video backbone checkpoint directory.",
    )
    parser.add_argument(
        "--smolvlm2-ckpt-path",
        type=str,
        default=os.environ.get("SMOLVLM2_CKPT_PATH", str(default_smolvlm2_ckpt())),
        help="Local SmolVLM2 checkpoint used by Mobile-OV.",
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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    output_dir = resolve_path(args.output_dir)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)

    model = MobileOVModel(
        generation_ckpt_path=args.checkpoint,
        video_backbone_checkpoint_dir=args.checkpoint_dir,
        smolvlm2_ckpt_path=args.smolvlm2_ckpt_path,
        device=args.device,
        dtype=args.dtype,
    )
    output_path = model.generate_video(
        prompt=args.prompt,
        output_dir=output_dir,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        steps=args.steps,
        cfg_scale=args.cfg_scale,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
