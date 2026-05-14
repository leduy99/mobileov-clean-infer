#!/usr/bin/env python3
"""
Minimal Mobile-OV generation path.

This file intentionally supports one architecture only:

SmolVLM2 text encoder
-> lexical-gated MCP bridge
-> SANA-video 2B 480p backbone
-> full DiT delta from the Mobile-OV checkpoint

It is written for readability, not experiment coverage.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SANA_REPO_ROOT = os.path.join(PROJECT_ROOT, "nets", "third_party", "sana")
if os.path.isdir(SANA_REPO_ROOT) and SANA_REPO_ROOT not in sys.path:
    sys.path.insert(0, SANA_REPO_ROOT)

from diffusion.data.datasets import utils as sana_dataset_utils
from diffusion.model.utils import prepare_prompt_ar
from nets.mobile_ov import MobileOVBridge


def _load_sana_runtime():
    from tools.inference import sana_video_runtime

    return sana_video_runtime


@dataclass(frozen=True)
class MobileOVSpec:
    projector_type: str = "mcp_lexical_gated"
    mcp_hidden_dim: int = 1536
    mcp_num_fuse_layers: int = 2
    mcp_use_refine: bool = True
    mcp_refine_kernel_size: int = 3
    mcp_lexical_bottleneck_dim: int = 256
    mcp_lexical_gate_init: float = 0.2
    strict_sana_parity_text_path: bool = True
    strict_sana_use_full_text_window: bool = True
    strict_sana_token_select_strategy: str = "head_uniform_tail"
    strict_sana_head_tokens: int = 96
    strict_sana_tail_tokens: int = 96
    fail_fast_mask: bool = True
    sana_model_max_length: int = 300
    student_max_length: int = 512
    caption_channels: int = 2304
    adapter_in_channels: int = 960
    adapter_out_channels: int = 2304
    adapter_query_length: int = 64
    adapter_num_encoder_layers: int = 2
    adapter_num_decoder_layers: int = 2
    adapter_ff_mult: int = 2
    resampler_num_heads: int = 8
    resampler_mlp_mult: int = 2
    smol_vh_num_queries: int = 1
    sampling_algo: str = "flow_dpm-solver"
    inference_flow_shift: float = 7.0


SPEC = MobileOVSpec()
TOKENIZER_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
SANA_CONFIG_PATH = "configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml"


def _get_base_ratios(config, height: int, width: int):
    image_size = getattr(getattr(config, "model", {}), "image_size", None) or height
    if getattr(config.vae, "vae_downsample_rate", 8) in [16, 32]:
        ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST_DIV32"
    else:
        ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST"
    base_ratios = getattr(sana_dataset_utils, ratio_name, None)
    if base_ratios is None:
        base_ratios = {f"{height / width:.2f}": [float(height), float(width)]}
    return base_ratios


def _normalize_prompt(prompt: str) -> str:
    return " ".join(str(prompt).strip().split())


def _pad_or_truncate_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
    cur_len = int(x.shape[1])
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[:, :target_len, :]
    pad_len = target_len - cur_len
    pad = torch.zeros((x.shape[0], pad_len, x.shape[2]), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=1)


def _pad_or_truncate_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
    cur_len = int(mask.shape[1])
    if cur_len == target_len:
        return mask
    if cur_len > target_len:
        return mask[:, :target_len]
    pad_len = target_len - cur_len
    pad = torch.zeros((mask.shape[0], pad_len), device=mask.device, dtype=mask.dtype)
    return torch.cat([mask, pad], dim=1)


def _load_checkpoint_bundle(bridge_ckpt: str):
    checkpoint = torch.load(bridge_ckpt, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Expected a dict checkpoint for Mobile-OV generation.")

    student_state = checkpoint.get("student_state", checkpoint)
    infer_hints = checkpoint.get("infer_hints", {}) or {}
    dit_state = checkpoint.get("dit_trainable_state", {}) or {}
    if not isinstance(student_state, dict):
        raise RuntimeError("Expected student_state to be a dict.")
    if not isinstance(infer_hints, dict):
        raise RuntimeError("Expected infer_hints to be a dict.")
    if not isinstance(dit_state, dict):
        raise RuntimeError("Expected dit_trainable_state to be a dict.")

    _assert_supported_checkpoint(student_state, infer_hints, dit_state)
    return student_state, dit_state


def _assert_supported_checkpoint(student_state: dict, infer_hints: dict, dit_state: dict) -> None:
    unsupported = []

    if infer_hints.get("projector_type", SPEC.projector_type) != SPEC.projector_type:
        unsupported.append(f"projector_type={infer_hints.get('projector_type')}")
    if bool(infer_hints.get("student_lora_enable", False)):
        unsupported.append("student_lora_enable=True")
    if bool(infer_hints.get("dit_lora_enable", False)):
        unsupported.append("dit_lora_enable=True")
    if bool(infer_hints.get("train_use_chi_prompt", False)):
        unsupported.append("train_use_chi_prompt=True")
    if bool(infer_hints.get("train_use_prompt_templates", False)):
        unsupported.append("train_use_prompt_templates=True")

    projector_state = student_state.get("projector", {})
    if not isinstance(projector_state, dict) or not projector_state:
        raise RuntimeError("This repo expects a projector state for the lexical-gated bridge.")
    if not dit_state:
        raise RuntimeError("This repo expects a non-empty dit_trainable_state for full-DiT inference.")

    if unsupported:
        raise RuntimeError(
            "Unsupported Mobile-OV checkpoint for the clean repo single-path runtime: "
            + ", ".join(unsupported)
        )


def _build_bridge(device: torch.device, dtype: torch.dtype, smolvlm2_ckpt_path: str) -> MobileOVBridge:
    return MobileOVBridge(
        smolvlm2_ckpt_path=smolvlm2_ckpt_path,
        adapter_ckpt_dir=None,
        adapter_in_channels=SPEC.adapter_in_channels,
        adapter_out_channels=SPEC.adapter_out_channels,
        adapter_query_length=SPEC.adapter_query_length,
        adapter_num_encoder_layers=SPEC.adapter_num_encoder_layers,
        adapter_num_decoder_layers=SPEC.adapter_num_decoder_layers,
        adapter_ff_mult=SPEC.adapter_ff_mult,
        smol_vh_num_queries=SPEC.smol_vh_num_queries,
        num_prompt_queries=SPEC.sana_model_max_length,
        caption_channels=SPEC.caption_channels,
        precision_dtype=dtype,
        device=device,
        tokenizer_model_id=TOKENIZER_MODEL_ID,
        force_adapter_query_length=SPEC.adapter_query_length,
        max_length=SPEC.student_max_length,
        use_vision_head=False,
        resampler_num_heads=SPEC.resampler_num_heads,
        resampler_mlp_mult=SPEC.resampler_mlp_mult,
        lora_enable=False,
        projector_type=SPEC.projector_type,
        mcp_hidden_dim=SPEC.mcp_hidden_dim,
        mcp_num_fuse_layers=SPEC.mcp_num_fuse_layers,
        mcp_use_refine=SPEC.mcp_use_refine,
        mcp_refine_kernel_size=SPEC.mcp_refine_kernel_size,
        mcp_fusion_temperature=1.0,
        mcp_lexical_bottleneck_dim=SPEC.mcp_lexical_bottleneck_dim,
        mcp_lexical_gate_init=SPEC.mcp_lexical_gate_init,
        strict_sana_parity_text_path=SPEC.strict_sana_parity_text_path,
        strict_sana_use_full_text_window=SPEC.strict_sana_use_full_text_window,
        strict_sana_token_select_strategy=SPEC.strict_sana_token_select_strategy,
        strict_sana_head_tokens=SPEC.strict_sana_head_tokens,
        strict_sana_tail_tokens=SPEC.strict_sana_tail_tokens,
        fail_fast_mask=SPEC.fail_fast_mask,
        sana_model_max_length=SPEC.sana_model_max_length,
        sana_chi_prompt="",
    )


def _load_checkpoint_weights(bridge: MobileOVBridge, projector_state: dict, diffusion_model, dit_state: dict) -> None:
    missing, unexpected = bridge.projector.load_state_dict(projector_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            f"Unexpected projector load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )

    missing, unexpected = diffusion_model.load_state_dict(dit_state, strict=False)
    loaded = max(0, len(dit_state) - len(unexpected))
    if loaded == 0:
        raise RuntimeError("No DiT delta keys were loaded from dit_trainable_state.")
    print(
        f"Loaded full-DiT delta from checkpoint: keys={len(dit_state)} loaded={loaded} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def _build_conditioning(
    bridge: MobileOVBridge,
    prompt_text: str,
    negative_prompt: str,
    cfg_scale: float,
    device: torch.device,
    dtype: torch.dtype,
):
    with torch.no_grad():
        cond_embeddings, cond_mask = bridge([prompt_text], return_mask=True)
        negative_embeddings = None
        negative_mask = None
        if cfg_scale > 1.0:
            negative_embeddings, negative_mask = bridge([negative_prompt], return_mask=True)

    if negative_embeddings is not None:
        target_len = max(int(cond_embeddings.shape[1]), int(negative_embeddings.shape[1]))
        cond_embeddings = _pad_or_truncate_seq(cond_embeddings, target_len)
        negative_embeddings = _pad_or_truncate_seq(negative_embeddings, target_len)
        cond_mask = _pad_or_truncate_mask(cond_mask.to(device=device, dtype=torch.long), target_len)
        negative_mask = _pad_or_truncate_mask(negative_mask.to(device=device, dtype=torch.long), target_len)
        batch_mask = torch.cat([negative_mask, cond_mask], dim=0)
    else:
        cond_mask = cond_mask.to(device=device, dtype=torch.long)
        batch_mask = cond_mask

    cond_embeddings = cond_embeddings.unsqueeze(1)
    if negative_embeddings is not None:
        negative_embeddings = negative_embeddings.unsqueeze(1)

    return cond_embeddings, negative_embeddings, batch_mask


def _save_output(video: np.ndarray, output_dir: str, prompt_text: str, runtime) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in prompt_text if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

    if video.shape[0] == 1:
        from PIL import Image

        png_path = os.path.join(output_dir, f"q1_student_{timestamp}_{safe_prompt[:40]}.png")
        Image.fromarray(video[0]).save(png_path)
        print(f"Saved image to: {png_path}")
        return

    out_path = os.path.join(output_dir, f"q1_student_{timestamp}_{safe_prompt[:40]}.mp4")
    runtime.save_video(video, out_path, fps=16)
    print(f"Saved video to: {out_path}")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Mobile-OV generator")
    parser.add_argument("--bridge-ckpt", type=str, required=True, help="Mobile-OV checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, default="omni_ckpts/sana_video_2b_480p")
    parser.add_argument(
        "--smolvlm2-ckpt-path",
        type=str,
        default=os.environ.get("SMOLVLM2_CKPT_PATH", "omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"),
    )
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    runtime = _load_sana_runtime()

    student_state, dit_state = _load_checkpoint_bundle(args.bridge_ckpt)
    projector_state = student_state["projector"]

    print(
        "Mobile-OV runtime: projector=%s hidden=%d K=%d refine=%s flow_shift=%.1f sampler=%s"
        % (
            SPEC.projector_type,
            SPEC.mcp_hidden_dim,
            SPEC.mcp_num_fuse_layers,
            SPEC.mcp_use_refine,
            SPEC.inference_flow_shift,
            SPEC.sampling_algo,
        )
    )
    print("Checkpoint contract validated for the clean single-path Mobile-OV runtime.")

    config = runtime.load_config_file(SANA_CONFIG_PATH)
    if not os.path.exists(args.checkpoint_dir) and hasattr(runtime, "download_checkpoint"):
        runtime.download_checkpoint(local_dir=args.checkpoint_dir)

    latent_size = args.height // config.vae.vae_downsample_rate
    models = runtime.load_sana_models(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
        model_dtype=dtype,
        vae_dtype=torch.float32,
        latent_size=latent_size,
        load_text_encoder=False,
    )

    prompt_clean, _, hw, _, _ = prepare_prompt_ar(
        _normalize_prompt(args.prompt),
        _get_base_ratios(config, args.height, args.width),
        device=device,
        show=False,
    )
    prompt_text = prompt_clean.strip()
    height, width = int(hw[0, 0].item()), int(hw[0, 1].item())

    bridge = _build_bridge(device=device, dtype=dtype, smolvlm2_ckpt_path=args.smolvlm2_ckpt_path)
    bridge_tokenizer = bridge._get_tokenizer()
    if type(bridge_tokenizer).__name__ == "SimpleTokenizer":
        raise RuntimeError(
            "SimpleTokenizer fallback detected. Please make sure the SmolVLM2 tokenizer cache is available."
        )

    _load_checkpoint_weights(bridge, projector_state, models["diffusion_model"], dit_state)

    text_embeddings, negative_embeddings, batch_mask = _build_conditioning(
        bridge=bridge,
        prompt_text=prompt_text,
        negative_prompt=args.negative_prompt,
        cfg_scale=float(args.cfg_scale),
        device=device,
        dtype=dtype,
    )

    vae_stride = getattr(config.vae, "vae_stride", [1, config.vae.vae_downsample_rate, config.vae.vae_downsample_rate])
    vae_stride_t = vae_stride[0] if isinstance(vae_stride, list) and len(vae_stride) >= 1 else 1
    latent_t = int(args.num_frames - 1) // int(vae_stride_t) + 1
    latent_shape = (
        1,
        int(config.vae.vae_latent_dim),
        latent_t,
        height // int(config.vae.vae_downsample_rate),
        width // int(config.vae.vae_downsample_rate),
    )

    hw_tensor = torch.tensor([[latent_shape[-2], latent_shape[-1]]], dtype=torch.float, device=device)
    model_kwargs = {
        "data_info": {"img_hw": hw_tensor},
        "mask": batch_mask.unsqueeze(1).unsqueeze(1),
    }

    print(
        "Sampling: frames=%d latent_t=%d steps=%d cfg=%.2f height=%d width=%d"
        % (int(args.num_frames), int(latent_t), int(args.steps), float(args.cfg_scale), int(height), int(width))
    )
    print(f"Prompt text: {prompt_text}")

    generator = torch.Generator(device=device).manual_seed(int(args.seed))
    latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    latents = runtime.flow_matching_sampling(
        models["diffusion_model"],
        latents,
        text_embeddings,
        negative_embeddings,
        num_steps=int(args.steps),
        device=str(device),
        cfg_scale=float(args.cfg_scale),
        flow_shift=SPEC.inference_flow_shift,
        model_kwargs=model_kwargs,
        sampling_algo=SPEC.sampling_algo,
    )

    latents = latents.to(models.get("vae_dtype", latents.dtype))
    video = models["vae"].decode(latents) if hasattr(models["vae"], "decode") else None
    if video is None:
        from diffusion.model.builder import vae_decode

        video = vae_decode(config.vae.vae_type, models["vae"], latents)
    if isinstance(video, list):
        video = torch.stack(video, dim=0)

    video = video[0].permute(1, 2, 3, 0).cpu().numpy()
    video = np.clip((video + 1.0) / 2.0, 0, 1)
    video = (video * 255).astype(np.uint8)

    _save_output(video, args.output_dir, prompt_text, runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
