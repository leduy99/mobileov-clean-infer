#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os

from nets.mobile_ov import MobileOVModel, default_smolvlm2_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mobile-OV understanding")
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=os.environ.get("SMOLVLM2_CKPT_PATH", str(default_smolvlm2_ckpt())),
        help="Local SmolVLM2 checkpoint (.pt).",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default=os.environ.get("SMOLVLM2_TOKENIZER_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
        help="Tokenizer/processor id used for text and media preprocessing.",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Question or instruction.")
    parser.add_argument("--image", type=str, default=None, help="Optional image path.")
    parser.add_argument("--video", type=str, default=None, help="Optional video path.")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames to sample from a video input.")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Model dtype.")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    model = MobileOVModel(
        smolvlm2_ckpt_path=args.ckpt_path,
        tokenizer_model_id=args.tokenizer_model_id,
        device=args.device,
        dtype=args.dtype,
    )
    text = model.understand(
        prompt=args.prompt,
        image_path=args.image,
        video_path=args.video,
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
