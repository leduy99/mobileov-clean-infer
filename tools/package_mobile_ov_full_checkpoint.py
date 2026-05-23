#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
from collections import OrderedDict
from pathlib import Path

import torch


FULL_CHECKPOINT_FORMAT = "mobile_ov_full_checkpoint_v1"
DEFAULT_TOKENIZER_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
TOKENIZER_ASSET_ALLOWLIST = {
    "added_tokens.json",
    "chat_template.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
}
MOBILE_OV_INFERENCE_KEYS = (
    "step",
    "micro_step",
    "student_state",
    "dit_train_modules",
    "infer_hints",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pack a complete Mobile-OV inference checkpoint: SmolVLM2, bridge, "
            "merged video DiT, and VAE in one .pt file."
        )
    )
    parser.add_argument("--mobile-ov-checkpoint", required=True, help="Mobile-OV training/inference checkpoint.")
    parser.add_argument("--smolvlm2-checkpoint", required=True, help="Converted SmolVLM2 checkpoint.")
    parser.add_argument("--video-backbone-checkpoint", required=True, help="Base SANA-video DiT checkpoint.")
    parser.add_argument("--vae-checkpoint", required=True, help="SANA-video VAE checkpoint.")
    parser.add_argument(
        "--tokenizer-assets-dir",
        default=None,
        help="Local SmolVLM2 tokenizer/processor directory. Defaults to the cached HF snapshot.",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        default=DEFAULT_TOKENIZER_MODEL_ID,
        help="HF id used only to locate cached tokenizer/processor assets when --tokenizer-assets-dir is omitted.",
    )
    parser.add_argument(
        "--allow-tokenizer-download",
        action="store_true",
        help="Allow downloading tokenizer/processor assets if they are not already cached.",
    )
    parser.add_argument("--output", required=True, help="Output full Mobile-OV checkpoint .pt path.")
    return parser.parse_args()


def source_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path).encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()[:16]


def load_dict_checkpoint(path: Path) -> dict:
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Expected a dict checkpoint: {path}")
    return checkpoint


def extract_diffusion_state(checkpoint: dict) -> OrderedDict:
    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint
    if not isinstance(state, dict):
        raise RuntimeError("Expected diffusion checkpoint to contain a state dict.")
    return OrderedDict((key.replace("module.", ""), value) for key, value in state.items())


def merge_dit_state(base_state: OrderedDict, delta_state: dict) -> OrderedDict:
    if not delta_state:
        raise RuntimeError("Mobile-OV checkpoint has empty dit_trainable_state; cannot build full DiT checkpoint.")

    missing = [key for key in delta_state if key not in base_state]
    if missing:
        raise RuntimeError(f"Delta has {len(missing)} keys missing from base DiT. First keys: {missing[:5]}")

    shape_mismatch = [
        (key, tuple(base_state[key].shape), tuple(value.shape))
        for key, value in delta_state.items()
        if hasattr(base_state[key], "shape")
        and hasattr(value, "shape")
        and tuple(base_state[key].shape) != tuple(value.shape)
    ]
    if shape_mismatch:
        raise RuntimeError(f"Delta/base shape mismatch. First mismatches: {shape_mismatch[:3]}")

    merged = OrderedDict(base_state)
    for key, value in delta_state.items():
        merged[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return merged


def slim_mobile_ov_checkpoint(checkpoint: dict) -> dict:
    slim = {key: checkpoint[key] for key in MOBILE_OV_INFERENCE_KEYS if key in checkpoint}
    slim["dit_trainable_state"] = {}
    slim["checkpoint_role"] = "full_checkpoint_metadata_only"
    return slim


def resolve_tokenizer_assets_dir(
    *,
    tokenizer_assets_dir: str | None,
    tokenizer_model_id: str,
    allow_download: bool,
) -> Path:
    if tokenizer_assets_dir:
        path = Path(tokenizer_assets_dir).expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            tokenizer_model_id,
            local_files_only=not allow_download,
            allow_patterns=sorted(TOKENIZER_ASSET_ALLOWLIST),
        )
    ).resolve()


def pack_tokenizer_assets(tokenizer_dir: Path) -> dict:
    files = {}
    for name in sorted(TOKENIZER_ASSET_ALLOWLIST):
        path = tokenizer_dir / name
        if path.is_file():
            files[name] = path.read_bytes()

    required = {"tokenizer.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json"}
    missing = sorted(name for name in required if name not in files)
    if missing:
        raise FileNotFoundError(f"Tokenizer/processor assets missing from {tokenizer_dir}: {missing}")

    digest = hashlib.sha256()
    for name, raw_bytes in files.items():
        digest.update(name.encode())
        digest.update(raw_bytes)
    return {
        "digest": digest.hexdigest()[:16],
        "source_dir": str(tokenizer_dir),
        "files": files,
    }


def main() -> int:
    args = parse_args()
    mobile_ov_path = Path(args.mobile_ov_checkpoint).expanduser().resolve()
    smolvlm2_path = Path(args.smolvlm2_checkpoint).expanduser().resolve()
    video_backbone_path = Path(args.video_backbone_checkpoint).expanduser().resolve()
    vae_path = Path(args.vae_checkpoint).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    for path in (mobile_ov_path, smolvlm2_path, video_backbone_path, vae_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    mobile_ov_checkpoint = load_dict_checkpoint(mobile_ov_path)
    video_backbone_checkpoint = load_dict_checkpoint(video_backbone_path)
    vae_checkpoint = torch.load(str(vae_path), map_location="cpu")
    if not isinstance(vae_checkpoint, dict):
        raise RuntimeError(f"Expected a dict VAE checkpoint: {vae_path}")

    base_dit_state = extract_diffusion_state(video_backbone_checkpoint)
    merged_dit_state = merge_dit_state(
        base_dit_state,
        mobile_ov_checkpoint.get("dit_trainable_state", {}),
    )
    tokenizer_assets_dir = resolve_tokenizer_assets_dir(
        tokenizer_assets_dir=args.tokenizer_assets_dir,
        tokenizer_model_id=args.tokenizer_model_id,
        allow_download=bool(args.allow_tokenizer_download),
    )
    tokenizer_assets = pack_tokenizer_assets(tokenizer_assets_dir)
    digest = source_digest([mobile_ov_path, smolvlm2_path, video_backbone_path, vae_path])

    full_checkpoint = {
        "format": FULL_CHECKPOINT_FORMAT,
        "mobile_ov_checkpoint": slim_mobile_ov_checkpoint(mobile_ov_checkpoint),
        "smolvlm2_checkpoint_bytes": smolvlm2_path.read_bytes(),
        "tokenizer_assets": tokenizer_assets,
        "video_backbone": {
            "digest": digest,
            "diffusion_filename": "checkpoints/SANA_Video_2B_480p.pth",
            "diffusion_checkpoint": {"state_dict": merged_dit_state},
            "vae_filename": "vae/Wan2.1_VAE.pth",
            "vae_checkpoint": vae_checkpoint,
        },
        "meta": {
            "mobile_ov_checkpoint": str(mobile_ov_path),
            "smolvlm2_checkpoint": str(smolvlm2_path),
            "video_backbone_checkpoint": str(video_backbone_path),
            "vae_checkpoint": str(vae_path),
            "tokenizer_assets_dir": str(tokenizer_assets_dir),
            "note": "Complete Mobile-OV inference checkpoint for conversion and clean inference.",
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(full_checkpoint, str(output_path))
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
