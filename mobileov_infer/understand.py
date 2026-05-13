from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import torch
from PIL import Image


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean Mobile-OV understanding wrapper")
    parser.add_argument(
        "--model-id",
        type=str,
        default="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        help="HF model id for SmolVLM2 understanding",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Question or instruction")
    parser.add_argument("--image", type=str, default=None, help="Optional image path")
    parser.add_argument("--video", type=str, default=None, help="Optional video path")
    parser.add_argument("--num-frames", type=int, default=8, help="Frames to sample from video")
    parser.add_argument("--max-new-tokens", type=int, default=128, help="Maximum generated tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device")
    return parser.parse_args()


def resolve_path(raw: Optional[str]) -> Optional[Path]:
    if raw is None:
        return None
    return Path(raw).expanduser().resolve()


def _sample_video_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_frames

    sample_count = max(1, int(num_frames))
    frame_indices = [round(i * max(total_frames - 1, 0) / max(sample_count - 1, 1)) for i in range(sample_count)]

    frames: List[Image.Image] = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    cap.release()
    if not frames:
        raise RuntimeError(f"Failed to extract frames from video: {video_path}")
    return frames


def _load_images(image_path: Optional[Path], video_path: Optional[Path], num_frames: int) -> Optional[List[Image.Image]]:
    if image_path is None and video_path is None:
        return None

    images: List[Image.Image] = []
    if image_path is not None:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        images.append(Image.open(image_path).convert("RGB"))

    if video_path is not None:
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        images.extend(_sample_video_frames(video_path, num_frames))

    return images if images else None


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from transformers import AutoModelForImageTextToText, AutoProcessor

    device = torch.device(args.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    logger.info("Loading processor from %s", args.model_id)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    logger.info("Loading model from %s", args.model_id)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    image_path = resolve_path(args.image)
    video_path = resolve_path(args.video)
    images = _load_images(image_path, video_path, args.num_frames)

    if images is not None:
        content = [{"type": "image"} for _ in images]
        content.append({"type": "text", "text": args.prompt})
        messages = [{"role": "user", "content": content}]
    else:
        messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]

    templated_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    processor_kwargs = {"text": templated_prompt, "return_tensors": "pt"}
    if images is not None:
        processor_kwargs["images"] = images
    inputs = processor(**processor_kwargs)
    model_dtype = next(model.parameters()).dtype
    normalized_inputs = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            normalized_inputs[key] = value
            continue
        if torch.is_floating_point(value):
            normalized_inputs[key] = value.to(device=device, dtype=model_dtype)
        else:
            normalized_inputs[key] = value.to(device=device)
    inputs = normalized_inputs

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
    }
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_length = inputs["input_ids"].shape[1]
    generated_ids = output_ids[:, prompt_length:]
    text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
