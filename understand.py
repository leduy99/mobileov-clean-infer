#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

import cv2
import torch
from PIL import Image

from nets.smolvlm2 import SmolVLMForConditionalGeneration, load_smolvlm2_from_ckpt


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parent


def default_smolvlm2_ckpt() -> Path:
    return REPO_ROOT / "omni_ckpts" / "smolvlm2_500m" / "smolvlm2_500m.pt"


def resolve_path(raw: Optional[str]) -> Optional[Path]:
    if raw is None:
        return None
    return Path(raw).expanduser().resolve()


def sample_video_frames(video_path: Path, num_frames: int) -> List[Image.Image]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = max(1, int(num_frames))

    sample_count = max(1, int(num_frames))
    if sample_count == 1:
        frame_indices = [0]
    else:
        frame_indices = [round(i * max(total_frames - 1, 0) / max(sample_count - 1, 1)) for i in range(sample_count)]

    frames: List[Image.Image] = []
    for frame_index in frame_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))

    capture.release()
    if not frames:
        raise RuntimeError(f"Failed to extract frames from video: {video_path}")
    return frames


def load_images(image_path: Optional[Path], video_path: Optional[Path], num_frames: int) -> Optional[List[Image.Image]]:
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
        images.extend(sample_video_frames(video_path, num_frames))
    return images


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
    return parser.parse_args()


def build_inputs(
    prompt: str,
    images: Optional[List[Image.Image]],
    tokenizer_model_id: str,
):
    from transformers import AutoProcessor, AutoTokenizer

    processor = None
    tokenizer = None

    try:
        processor = AutoProcessor.from_pretrained(tokenizer_model_id, trust_remote_code=True)
        LOGGER.info("Loaded processor from %s", tokenizer_model_id)
    except Exception as exc:
        if images is not None:
            raise RuntimeError(
                "Failed to load AutoProcessor for multimodal understanding. "
                "The local SmolVLM2 model path is ready, but media preprocessing still needs the HF processor."
            ) from exc
        LOGGER.warning("Falling back to AutoTokenizer because AutoProcessor could not be loaded: %s", exc)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_id, trust_remote_code=True)

    if processor is not None:
        if images is not None:
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
        messages = [{"role": "user", "content": content}]
        templated_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        processor_kwargs = {"text": templated_prompt, "return_tensors": "pt"}
        if images is not None:
            processor_kwargs["images"] = images
        inputs = processor(**processor_kwargs)
        return inputs, processor, getattr(processor, "tokenizer", None)

    assert tokenizer is not None
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    return inputs, None, tokenizer


def move_inputs_to_device(inputs, device: torch.device, model_dtype: torch.dtype):
    normalized = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            normalized[key] = value
            continue
        if torch.is_floating_point(value):
            normalized[key] = value.to(device=device, dtype=model_dtype)
        else:
            normalized[key] = value.to(device=device)
    return normalized


def main() -> int:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    ckpt_path = resolve_path(args.ckpt_path)
    if ckpt_path is None or not ckpt_path.exists():
        raise FileNotFoundError(f"SmolVLM2 checkpoint not found: {ckpt_path}")

    device = torch.device(args.device)
    LOGGER.info("Loading local SmolVLM2 model from %s", ckpt_path)
    model = load_smolvlm2_from_ckpt(
        str(ckpt_path),
        device=device,
        model_class=SmolVLMForConditionalGeneration,
    )
    model.eval()

    image_path = resolve_path(args.image)
    video_path = resolve_path(args.video)
    images = load_images(image_path, video_path, args.num_frames)
    inputs, processor, tokenizer = build_inputs(args.prompt, images, args.tokenizer_model_id)

    model_dtype = next(model.parameters()).dtype
    inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)

    pad_token_id = None
    eos_token_id = None
    if tokenizer is not None:
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "do_sample": args.temperature > 0,
    }
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p
    if pad_token_id is not None:
        gen_kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        gen_kwargs["eos_token_id"] = eos_token_id

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_length = int(inputs["input_ids"].shape[1])
    generated_ids = output_ids[:, prompt_length:]
    if tokenizer is not None:
        text = tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
    elif processor is not None and hasattr(processor, "batch_decode"):
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    else:
        raise RuntimeError("No tokenizer available to decode model output.")

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
