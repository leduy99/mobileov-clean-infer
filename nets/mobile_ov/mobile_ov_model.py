"""
Full Mobile-OV inference model.

This file is the single place that wires together the active clean-repo
architecture:

Prompt / media
  -> local SmolVLM2 understanding model
  -> Mobile-OV lexical-gated bridge
  -> vendored video diffusion backbone

The goal is readability. The clean repo intentionally supports one checkpoint
family and one architecture only.
"""

from __future__ import annotations

import logging
import os
import sys
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


LOGGER = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
VIDEO_BACKBONE_REPO_ROOT = REPO_ROOT / "nets" / "third_party" / "video_backbone"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if VIDEO_BACKBONE_REPO_ROOT.is_dir() and str(VIDEO_BACKBONE_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(VIDEO_BACKBONE_REPO_ROOT))

from nets.smolvlm2 import SmolVLMForConditionalGeneration, load_smolvlm2_from_ckpt

if TYPE_CHECKING:
    from nets.mobile_ov.mobile_ov_bridge import MobileOVBridge


TOKENIZER_MODEL_ID = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VIDEO_BACKBONE_CONFIG_PATH = "configs/video_backbone_config/Sana_2000M_480px_AdamW_fsdp.yaml"
MOBILE_OV_BUNDLE_FORMAT = "mobile_ov_bundle_v1"
MOBILE_OV_FULL_CHECKPOINT_FORMAT = "mobile_ov_full_checkpoint_v1"


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


def resolve_path(raw: str | Path | None) -> Path | None:
    if raw is None:
        return None
    return Path(raw).expanduser().resolve()


def default_generation_ckpt() -> Path:
    return (
        REPO_ROOT
        / "omni_ckpts"
        / "hf_mobile_ov"
        / "mobile_ov_135k_full.pt"
    )


def default_video_backbone_checkpoint_dir() -> Path:
    return REPO_ROOT / "omni_ckpts" / "sana_video_2b_480p"


def default_smolvlm2_ckpt() -> Path:
    return REPO_ROOT / "omni_ckpts" / "smolvlm2_500m" / "smolvlm2_500m.pt"


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


class MobileOVModel(nn.Module):
    """
    Full clean-repo Mobile-OV inference model.

    This class exposes the two user-facing capabilities of the current system:
    generation and understanding.
    """

    def __init__(
        self,
        *,
        generation_ckpt_path: str | Path | None = None,
        video_backbone_checkpoint_dir: str | Path | None = None,
        smolvlm2_ckpt_path: str | Path | None = None,
        tokenizer_model_id: str = TOKENIZER_MODEL_ID,
        device: str = "cuda:0",
        dtype: str = "bf16",
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.model_dtype = torch.bfloat16 if dtype == "bf16" else torch.float32
        self.tokenizer_model_id = tokenizer_model_id

        self.generation_ckpt_path = resolve_path(generation_ckpt_path) or default_generation_ckpt().resolve()
        self.video_backbone_checkpoint_dir = (
            resolve_path(video_backbone_checkpoint_dir) or default_video_backbone_checkpoint_dir().resolve()
        )
        self.smolvlm2_ckpt_path = resolve_path(smolvlm2_ckpt_path) or default_smolvlm2_ckpt().resolve()
        self._embedded_smolvlm2_ckpt_path: Path | None = None
        self._uses_full_checkpoint = False

        self.bridge: "MobileOVBridge | None" = None
        self.video_diffusion_model: nn.Module | None = None
        self.video_vae: nn.Module | None = None
        self.understanding_model: SmolVLMForConditionalGeneration | None = None

        self._runtime = None
        self._video_config = None

    def load_generation(self) -> "MobileOVModel":
        if self.bridge is not None and self.video_diffusion_model is not None and self.video_vae is not None:
            return self

        if not self.generation_ckpt_path.exists():
            raise FileNotFoundError(f"Generation checkpoint not found: {self.generation_ckpt_path}")

        student_state, dit_state = self._load_checkpoint_bundle(self.generation_ckpt_path)
        if not self.video_backbone_checkpoint_dir.is_dir():
            raise FileNotFoundError(
                f"Video backbone checkpoint directory not found: {self.video_backbone_checkpoint_dir}"
            )

        self._runtime = self._load_video_backbone_runtime()
        self._ensure_smolvlm2_checkpoint_available()
        projector_state = student_state["projector"]

        self._video_config = self._runtime.load_config_file(VIDEO_BACKBONE_CONFIG_PATH)
        latent_size = 480 // self._video_config.vae.vae_downsample_rate
        models = self._runtime.load_backbone_models(
            config=self._video_config,
            checkpoint_dir=str(self.video_backbone_checkpoint_dir),
            device=str(self.device),
            model_dtype=self.model_dtype,
            vae_dtype=torch.float32,
            latent_size=latent_size,
            load_text_encoder=False,
        )

        self.bridge = self._build_bridge()
        self._assert_real_bridge_tokenizer(self.bridge)
        self._load_checkpoint_weights(
            bridge=self.bridge,
            projector_state=projector_state,
            diffusion_model=models["diffusion_model"],
            dit_state=dit_state,
        )

        self.video_diffusion_model = models["diffusion_model"]
        self.video_vae = models["vae"]
        self.eval()
        return self

    def load_understanding(self) -> "MobileOVModel":
        if self.understanding_model is not None:
            return self

        self._ensure_smolvlm2_checkpoint_available()

        self.understanding_model = load_smolvlm2_from_ckpt(
            str(self.smolvlm2_ckpt_path),
            device=self.device,
            model_class=SmolVLMForConditionalGeneration,
        )
        self.understanding_model.eval()
        self.eval()
        return self

    def generate_video(
        self,
        *,
        prompt: str,
        output_dir: str | Path,
        num_frames: int = 81,
        height: int = 480,
        width: int = 832,
        steps: int = 24,
        cfg_scale: float = 6.0,
        negative_prompt: str = "",
        seed: int = 0,
    ) -> Path:
        self.load_generation()
        assert self.bridge is not None
        assert self.video_diffusion_model is not None
        assert self.video_vae is not None
        assert self._runtime is not None
        assert self._video_config is not None
        from diffusion.model.utils import prepare_prompt_ar

        prompt_clean, _, hw, _, _ = prepare_prompt_ar(
            _normalize_prompt(prompt),
            self._get_base_ratios(self._video_config, height, width),
            device=self.device,
            show=False,
        )
        prompt_text = prompt_clean.strip()
        height, width = int(hw[0, 0].item()), int(hw[0, 1].item())

        text_embeddings, negative_embeddings, batch_mask = self._build_conditioning(
            prompt_text=prompt_text,
            negative_prompt=negative_prompt,
            cfg_scale=float(cfg_scale),
        )

        vae_stride = getattr(
            self._video_config.vae,
            "vae_stride",
            [1, self._video_config.vae.vae_downsample_rate, self._video_config.vae.vae_downsample_rate],
        )
        vae_stride_t = vae_stride[0] if isinstance(vae_stride, list) and len(vae_stride) >= 1 else 1
        latent_t = int(num_frames - 1) // int(vae_stride_t) + 1
        latent_shape = (
            1,
            int(self._video_config.vae.vae_latent_dim),
            latent_t,
            height // int(self._video_config.vae.vae_downsample_rate),
            width // int(self._video_config.vae.vae_downsample_rate),
        )

        hw_tensor = torch.tensor([[latent_shape[-2], latent_shape[-1]]], dtype=torch.float, device=self.device)
        model_kwargs = {
            "data_info": {"img_hw": hw_tensor},
            "mask": batch_mask.unsqueeze(1).unsqueeze(1),
        }

        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        latents = torch.randn(latent_shape, device=self.device, dtype=self.model_dtype, generator=generator)
        latents = self._runtime.flow_matching_sampling(
            self.video_diffusion_model,
            latents,
            text_embeddings,
            negative_embeddings,
            num_steps=int(steps),
            device=str(self.device),
            cfg_scale=float(cfg_scale),
            flow_shift=SPEC.inference_flow_shift,
            model_kwargs=model_kwargs,
            sampling_algo=SPEC.sampling_algo,
        )

        vae_dtype = torch.float32
        latents = latents.to(vae_dtype)
        video = self.video_vae.decode(latents) if hasattr(self.video_vae, "decode") else None
        if video is None:
            from diffusion.model.builder import vae_decode

            video = vae_decode(self._video_config.vae.vae_type, self.video_vae, latents)
        if isinstance(video, list):
            video = torch.stack(video, dim=0)

        video = video[0].permute(1, 2, 3, 0).cpu().numpy()
        video = np.clip((video + 1.0) / 2.0, 0, 1)
        video = (video * 255).astype(np.uint8)
        return self._save_output(video=video, output_dir=output_dir, prompt_text=prompt_text)

    def generate_image(
        self,
        *,
        prompt: str,
        output_dir: str | Path,
        height: int = 480,
        width: int = 832,
        steps: int = 24,
        cfg_scale: float = 6.0,
        negative_prompt: str = "",
        seed: int = 0,
    ) -> Path:
        return self.generate_video(
            prompt=prompt,
            output_dir=output_dir,
            num_frames=1,
            height=height,
            width=width,
            steps=steps,
            cfg_scale=cfg_scale,
            negative_prompt=negative_prompt,
            seed=seed,
        )

    def understand(
        self,
        *,
        prompt: str,
        image_path: str | Path | None = None,
        video_path: str | Path | None = None,
        num_frames: int = 8,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float | None = None,
    ) -> str:
        self.load_understanding()
        assert self.understanding_model is not None

        images = self._load_media(image_path=image_path, video_path=video_path, num_frames=num_frames)
        inputs, processor, tokenizer = self._build_understanding_inputs(prompt=prompt, images=images)
        inputs = self._move_inputs_to_device(inputs)

        pad_token_id = tokenizer.pad_token_id if tokenizer is not None else None
        eos_token_id = tokenizer.eos_token_id if tokenizer is not None else None
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
        }
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if pad_token_id is not None:
            gen_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            gen_kwargs["eos_token_id"] = eos_token_id

        with torch.no_grad():
            output_ids = self.understanding_model.generate(**inputs, **gen_kwargs)

        prompt_length = int(inputs["input_ids"].shape[1])
        generated_ids = output_ids[:, prompt_length:]
        if tokenizer is not None:
            return tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        if processor is not None and hasattr(processor, "batch_decode"):
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        raise RuntimeError("No tokenizer available to decode model output.")

    def understand_text(
        self,
        *,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float | None = None,
    ) -> str:
        return self.understand(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def understand_image(
        self,
        *,
        image_path: str | Path,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float | None = None,
    ) -> str:
        return self.understand(
            prompt=prompt,
            image_path=image_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def understand_video(
        self,
        *,
        video_path: str | Path,
        prompt: str,
        num_frames: int = 8,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float | None = None,
    ) -> str:
        return self.understand(
            prompt=prompt,
            video_path=video_path,
            num_frames=num_frames,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    def _load_video_backbone_runtime(self):
        from tools.inference import video_backbone_runtime

        return video_backbone_runtime

    def _load_checkpoint_bundle(self, bridge_ckpt: Path):
        checkpoint = torch.load(str(bridge_ckpt), map_location="cpu")
        if not isinstance(checkpoint, dict):
            raise RuntimeError("Expected a dict checkpoint for Mobile-OV generation.")
        checkpoint = self._unwrap_mobile_ov_bundle(checkpoint, bridge_ckpt, materialize_video_backbone=True)

        student_state = checkpoint.get("student_state", checkpoint)
        infer_hints = checkpoint.get("infer_hints", {}) or {}
        dit_state = checkpoint.get("dit_trainable_state", {}) or {}
        if not isinstance(student_state, dict):
            raise RuntimeError("Expected student_state to be a dict.")
        if not isinstance(infer_hints, dict):
            raise RuntimeError("Expected infer_hints to be a dict.")
        if not isinstance(dit_state, dict):
            raise RuntimeError("Expected dit_trainable_state to be a dict.")

        self._assert_supported_checkpoint(student_state, infer_hints, dit_state)
        return student_state, dit_state

    def _unwrap_mobile_ov_bundle(
        self,
        checkpoint: dict,
        bundle_path: Path,
        *,
        materialize_video_backbone: bool,
    ) -> dict:
        checkpoint_format = checkpoint.get("format")
        if checkpoint_format not in {MOBILE_OV_BUNDLE_FORMAT, MOBILE_OV_FULL_CHECKPOINT_FORMAT}:
            return checkpoint

        mobile_ov_checkpoint = checkpoint.get("mobile_ov_checkpoint")
        if not isinstance(mobile_ov_checkpoint, dict):
            raise RuntimeError("Mobile-OV checkpoint bundle is missing a dict 'mobile_ov_checkpoint'.")

        smolvlm2_bytes = checkpoint.get("smolvlm2_checkpoint_bytes")
        if not isinstance(smolvlm2_bytes, (bytes, bytearray)):
            raise RuntimeError("Mobile-OV checkpoint bundle is missing embedded 'smolvlm2_checkpoint_bytes'.")

        tokenizer_assets = checkpoint.get("tokenizer_assets")
        if isinstance(tokenizer_assets, dict):
            tokenizer_dir = self._materialize_embedded_tokenizer_assets(
                tokenizer_assets,
                bundle_path=bundle_path,
            )
            self.tokenizer_model_id = str(tokenizer_dir)
            LOGGER.info("Using tokenizer/processor assets embedded in checkpoint: %s", tokenizer_dir)

        self._embedded_smolvlm2_ckpt_path = self._materialize_embedded_smolvlm2(
            bytes(smolvlm2_bytes),
            bundle_path=bundle_path,
        )
        self.smolvlm2_ckpt_path = self._embedded_smolvlm2_ckpt_path
        LOGGER.info("Using SmolVLM2 checkpoint embedded in bundle: %s", self.smolvlm2_ckpt_path)

        if checkpoint_format == MOBILE_OV_FULL_CHECKPOINT_FORMAT and materialize_video_backbone:
            self.video_backbone_checkpoint_dir = self._materialize_embedded_video_backbone(
                checkpoint,
                bundle_path=bundle_path,
            )
            self._uses_full_checkpoint = True
            LOGGER.info("Using video backbone embedded in full Mobile-OV checkpoint: %s", self.video_backbone_checkpoint_dir)

        return mobile_ov_checkpoint

    def _materialize_embedded_smolvlm2(self, checkpoint_bytes: bytes, *, bundle_path: Path) -> Path:
        digest = hashlib.sha256(checkpoint_bytes).hexdigest()[:16]
        cache_root = Path(os.environ.get("MOBILEOV_BUNDLE_CACHE", Path.home() / ".cache" / "mobile_ov" / "bundles"))
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / f"{bundle_path.stem}_{digest}_smolvlm2.pt"
        if target.exists() and target.stat().st_size == len(checkpoint_bytes):
            return target

        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(checkpoint_bytes)
        tmp.replace(target)
        return target

    def _materialize_embedded_video_backbone(self, checkpoint: dict, *, bundle_path: Path) -> Path:
        video_backbone = checkpoint.get("video_backbone")
        if not isinstance(video_backbone, dict):
            raise RuntimeError("Full Mobile-OV checkpoint is missing a dict 'video_backbone'.")

        diffusion_checkpoint = video_backbone.get("diffusion_checkpoint")
        vae_checkpoint = video_backbone.get("vae_checkpoint")
        if not isinstance(diffusion_checkpoint, dict):
            raise RuntimeError("Full Mobile-OV checkpoint is missing 'video_backbone.diffusion_checkpoint'.")
        if not isinstance(vae_checkpoint, dict):
            raise RuntimeError("Full Mobile-OV checkpoint is missing 'video_backbone.vae_checkpoint'.")

        digest = str(video_backbone.get("digest") or hashlib.sha256(str(bundle_path).encode()).hexdigest()[:16])
        cache_root = Path(os.environ.get("MOBILEOV_BUNDLE_CACHE", Path.home() / ".cache" / "mobile_ov" / "bundles"))
        target_dir = cache_root / f"{bundle_path.stem}_{digest}_video_backbone"
        diffusion_rel = Path(str(video_backbone.get("diffusion_filename", "checkpoints/SANA_Video_2B_480p.pth")))
        vae_rel = Path(str(video_backbone.get("vae_filename", "vae/Wan2.1_VAE.pth")))
        diffusion_path = target_dir / diffusion_rel
        vae_path = target_dir / vae_rel
        manifest_path = target_dir / ".mobile_ov_full_checkpoint.json"

        expected_manifest = {
            "format": MOBILE_OV_FULL_CHECKPOINT_FORMAT,
            "digest": digest,
            "diffusion_filename": str(diffusion_rel),
            "vae_filename": str(vae_rel),
        }
        if diffusion_path.exists() and vae_path.exists() and manifest_path.exists():
            try:
                if json.loads(manifest_path.read_text()) == expected_manifest:
                    return target_dir
            except json.JSONDecodeError:
                pass

        diffusion_path.parent.mkdir(parents=True, exist_ok=True)
        vae_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(diffusion_checkpoint, str(diffusion_path))
        torch.save(vae_checkpoint, str(vae_path))
        manifest_path.write_text(json.dumps(expected_manifest, indent=2, sort_keys=True))
        return target_dir

    def _materialize_embedded_tokenizer_assets(self, tokenizer_assets: dict, *, bundle_path: Path) -> Path:
        files = tokenizer_assets.get("files")
        if not isinstance(files, dict) or not files:
            raise RuntimeError("Embedded tokenizer assets are missing a non-empty 'files' dict.")

        digest = str(tokenizer_assets.get("digest") or hashlib.sha256(str(sorted(files)).encode()).hexdigest()[:16])
        cache_root = Path(os.environ.get("MOBILEOV_BUNDLE_CACHE", Path.home() / ".cache" / "mobile_ov" / "bundles"))
        target_dir = cache_root / f"{bundle_path.stem}_{digest}_tokenizer"
        manifest_path = target_dir / ".mobile_ov_tokenizer_assets.json"
        expected_manifest = {
            "format": MOBILE_OV_FULL_CHECKPOINT_FORMAT,
            "digest": digest,
            "files": sorted(files.keys()),
        }
        if target_dir.exists() and manifest_path.exists():
            try:
                if json.loads(manifest_path.read_text()) == expected_manifest:
                    return target_dir
            except json.JSONDecodeError:
                pass

        target_dir.mkdir(parents=True, exist_ok=True)
        for relative_name, raw_bytes in files.items():
            if not isinstance(raw_bytes, (bytes, bytearray)):
                raise RuntimeError(f"Tokenizer asset {relative_name!r} is not stored as bytes.")
            relative_path = Path(str(relative_name))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise RuntimeError(f"Unsafe tokenizer asset path: {relative_name!r}")
            output_path = target_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(bytes(raw_bytes))
        manifest_path.write_text(json.dumps(expected_manifest, indent=2, sort_keys=True))
        return target_dir

    def _ensure_smolvlm2_checkpoint_available(self) -> None:
        if self.smolvlm2_ckpt_path.exists():
            return
        if self.generation_ckpt_path.exists():
            checkpoint = torch.load(str(self.generation_ckpt_path), map_location="cpu")
            if isinstance(checkpoint, dict) and checkpoint.get("format") in {
                MOBILE_OV_BUNDLE_FORMAT,
                MOBILE_OV_FULL_CHECKPOINT_FORMAT,
            }:
                self._unwrap_mobile_ov_bundle(
                    checkpoint,
                    self.generation_ckpt_path,
                    materialize_video_backbone=False,
                )
                return
        raise FileNotFoundError(f"SmolVLM2 checkpoint not found: {self.smolvlm2_ckpt_path}")

    def _assert_supported_checkpoint(self, student_state: dict, infer_hints: dict, dit_state: dict) -> None:
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
        if not dit_state and not self._uses_full_checkpoint:
            raise RuntimeError("This repo expects a non-empty dit_trainable_state for full-DiT inference.")
        if unsupported:
            raise RuntimeError(
                "Unsupported Mobile-OV checkpoint for the clean single-path runtime: "
                + ", ".join(unsupported)
            )

    def _build_bridge(self) -> "MobileOVBridge":
        from nets.mobile_ov.mobile_ov_bridge import MobileOVBridge

        return MobileOVBridge(
            smolvlm2_ckpt_path=str(self.smolvlm2_ckpt_path),
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
            precision_dtype=self.model_dtype,
            device=self.device,
            tokenizer_model_id=self.tokenizer_model_id,
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

    def _assert_real_bridge_tokenizer(self, bridge: MobileOVBridge) -> None:
        tokenizer = bridge._get_tokenizer()
        if type(tokenizer).__name__ == "SimpleTokenizer":
            raise RuntimeError(
                "SimpleTokenizer fallback detected. Please make sure the SmolVLM2 tokenizer cache is available."
            )

    def _load_checkpoint_weights(
        self,
        *,
        bridge: "MobileOVBridge",
        projector_state: dict,
        diffusion_model: nn.Module,
        dit_state: dict,
    ) -> None:
        missing, unexpected = bridge.projector.load_state_dict(projector_state, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Unexpected projector load mismatch: missing={len(missing)} unexpected={len(unexpected)}"
            )

        if self._uses_full_checkpoint and not dit_state:
            LOGGER.info("Full Mobile-OV checkpoint already contains merged DiT weights; skipping delta load.")
            return

        missing, unexpected = diffusion_model.load_state_dict(dit_state, strict=False)
        loaded = max(0, len(dit_state) - len(unexpected))
        if loaded == 0:
            raise RuntimeError("No DiT delta keys were loaded from dit_trainable_state.")
        LOGGER.info(
            "Loaded full-DiT delta: keys=%d loaded=%d missing=%d unexpected=%d",
            len(dit_state),
            loaded,
            len(missing),
            len(unexpected),
        )

    def _build_conditioning(
        self,
        *,
        prompt_text: str,
        negative_prompt: str,
        cfg_scale: float,
    ):
        assert self.bridge is not None
        with torch.no_grad():
            cond_embeddings, cond_mask = self.bridge([prompt_text], return_mask=True)
            negative_embeddings = None
            negative_mask = None
            if cfg_scale > 1.0:
                negative_embeddings, negative_mask = self.bridge([negative_prompt], return_mask=True)

        if negative_embeddings is not None:
            target_len = max(int(cond_embeddings.shape[1]), int(negative_embeddings.shape[1]))
            cond_embeddings = _pad_or_truncate_seq(cond_embeddings, target_len)
            negative_embeddings = _pad_or_truncate_seq(negative_embeddings, target_len)
            cond_mask = _pad_or_truncate_mask(cond_mask.to(device=self.device, dtype=torch.long), target_len)
            negative_mask = _pad_or_truncate_mask(
                negative_mask.to(device=self.device, dtype=torch.long),
                target_len,
            )
            batch_mask = torch.cat([negative_mask, cond_mask], dim=0)
        else:
            cond_mask = cond_mask.to(device=self.device, dtype=torch.long)
            batch_mask = cond_mask

        cond_embeddings = cond_embeddings.unsqueeze(1)
        if negative_embeddings is not None:
            negative_embeddings = negative_embeddings.unsqueeze(1)
        return cond_embeddings, negative_embeddings, batch_mask

    def _get_base_ratios(self, config, height: int, width: int):
        from diffusion.data.datasets import utils as backbone_dataset_utils

        image_size = getattr(getattr(config, "model", {}), "image_size", None) or height
        if getattr(config.vae, "vae_downsample_rate", 8) in [16, 32]:
            ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST_DIV32"
        else:
            ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST"
        base_ratios = getattr(backbone_dataset_utils, ratio_name, None)
        if base_ratios is None:
            base_ratios = {f"{height / width:.2f}": [float(height), float(width)]}
        return base_ratios

    def _save_output(self, *, video: np.ndarray, output_dir: str | Path, prompt_text: str) -> Path:
        output_dir = resolve_path(output_dir)
        assert output_dir is not None
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c for c in prompt_text if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")

        if video.shape[0] == 1:
            png_path = output_dir / f"q1_student_{timestamp}_{safe_prompt[:40]}.png"
            Image.fromarray(video[0]).save(png_path)
            LOGGER.info("Saved image to %s", png_path)
            return png_path

        mp4_path = output_dir / f"q1_student_{timestamp}_{safe_prompt[:40]}.mp4"
        assert self._runtime is not None
        self._runtime.save_video(video, str(mp4_path), fps=16)
        LOGGER.info("Saved video to %s", mp4_path)
        return mp4_path

    def _load_media(
        self,
        *,
        image_path: str | Path | None,
        video_path: str | Path | None,
        num_frames: int,
    ) -> List[Image.Image] | None:
        resolved_image = resolve_path(image_path)
        resolved_video = resolve_path(video_path)
        if resolved_image is None and resolved_video is None:
            return None

        images: List[Image.Image] = []
        if resolved_image is not None:
            if not resolved_image.exists():
                raise FileNotFoundError(f"Image not found: {resolved_image}")
            images.append(Image.open(resolved_image).convert("RGB"))
        if resolved_video is not None:
            if not resolved_video.exists():
                raise FileNotFoundError(f"Video not found: {resolved_video}")
            images.extend(self.sample_video_frames(resolved_video, num_frames))
        return images

    def _build_understanding_inputs(self, *, prompt: str, images: List[Image.Image] | None):
        from transformers import AutoProcessor, AutoTokenizer

        processor = None
        tokenizer = None
        try:
            processor = AutoProcessor.from_pretrained(self.tokenizer_model_id, trust_remote_code=True)
            LOGGER.info("Loaded processor from %s", self.tokenizer_model_id)
        except Exception as exc:
            if images is not None:
                raise RuntimeError(
                    "Failed to load AutoProcessor for multimodal understanding. "
                    "The local SmolVLM2 model path is ready, but media preprocessing still needs the HF processor."
                ) from exc
            LOGGER.warning("Falling back to AutoTokenizer because AutoProcessor could not be loaded: %s", exc)
            tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model_id, trust_remote_code=True)

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

    def _move_inputs_to_device(self, inputs):
        understanding_dtype = self.model_dtype
        if self.understanding_model is not None:
            understanding_dtype = next(self.understanding_model.parameters()).dtype
        normalized = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                normalized[key] = value
                continue
            if torch.is_floating_point(value):
                normalized[key] = value.to(device=self.device, dtype=understanding_dtype)
            else:
                normalized[key] = value.to(device=self.device)
        return normalized

    @staticmethod
    def sample_video_frames(video_path: str | Path, num_frames: int) -> List[Image.Image]:
        resolved = resolve_path(video_path)
        assert resolved is not None
        capture = cv2.VideoCapture(str(resolved))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {resolved}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = max(1, int(num_frames))

        sample_count = max(1, int(num_frames))
        if sample_count == 1:
            frame_indices = [0]
        else:
            frame_indices = [
                round(i * max(total_frames - 1, 0) / max(sample_count - 1, 1))
                for i in range(sample_count)
            ]

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
            raise RuntimeError(f"Failed to extract frames from video: {resolved}")
        return frames
