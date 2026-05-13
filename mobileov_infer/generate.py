from __future__ import annotations

import argparse
import os

from .common import (
    default_generation_ckpt,
    resolve_path,
    run_backend_python,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Mobile-OV generation wrapper")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Generation checkpoint (.pt). Defaults to MOBILEOV_GENERATION_CKPT or the local 60k joint checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.environ.get("MOBILEOV_SANA_CHECKPOINT_DIR", "omni_ckpts/sana_video_2b_480p"),
        help="Local SANA checkpoint directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=os.environ.get("MOBILEOV_SANA_CONFIG", "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml"),
        help="SANA config path",
    )
    parser.add_argument(
        "--smolvlm2-ckpt-path",
        type=str,
        default=os.environ.get("SMOLVLM2_CKPT_PATH", "omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"),
        help="SmolVLM2 checkpoint path",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default=os.environ.get("SMOLVLM2_TOKENIZER_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
        help="Tokenizer/model id for the bridge text path",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt text")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    parser.add_argument("--num-frames", type=int, default=81, help="Number of frames")
    parser.add_argument("--height", type=int, default=480, help="Output height")
    parser.add_argument("--width", type=int, default=832, help="Output width")
    parser.add_argument("--steps", type=int, default=24, help="Sampling steps")
    parser.add_argument("--cfg-scale", type=float, default=6.0, help="CFG scale")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"], help="Inference dtype")
    parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--sana-backend", type=str, default="fixed", choices=["fixed", "legacy"], help="SANA backend")
    parser.add_argument("--sampling-algo", type=str, default=None, help="Optional sampling override")
    parser.add_argument("--motion-score", type=int, default=10, help="Motion score appended by backend")
    parser.add_argument("--use-chi-prompt", action="store_true", help="Enable CHI prompt mode")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ckpt_raw = args.checkpoint or os.environ.get("MOBILEOV_GENERATION_CKPT")
    checkpoint = resolve_path(ckpt_raw) if ckpt_raw else default_generation_ckpt().resolve()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint.exists():
        raise FileNotFoundError(f"Generation checkpoint not found: {checkpoint}")

    backend_args = [
        "--bridge-ckpt",
        str(checkpoint),
        "--checkpoint-dir",
        args.checkpoint_dir,
        "--config",
        args.config,
        "--smolvlm2-ckpt-path",
        args.smolvlm2_ckpt_path,
        "--tokenizer-model-id",
        args.tokenizer_model_id,
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
        "--sana-backend",
        args.sana_backend,
        "--motion-score",
        str(args.motion_score),
    ]
    if args.sampling_algo:
        backend_args.extend(["--sampling-algo", args.sampling_algo])
    if args.use_chi_prompt:
        backend_args.append("--use-chi-prompt")

    return run_backend_python(
        script_relpath="tools/inference/test_q1_student_video.py",
        argv=backend_args,
    )


if __name__ == "__main__":
    raise SystemExit(main())
