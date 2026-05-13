#!/usr/bin/env python3
"""
Test SANA video generation using Q1 student prompt embeddings (SmolVLM2 -> bridge).

Quick path:
1) Load SANA backbone (text encoder/VAE/DiT).
2) Load student checkpoint (bridge + SmolVLM2 LoRA + optional DiT LoRA).
3) Build student prompt embeddings, then run SANA flow-matching sampling.
4) Decode latents with WAN VAE and save MP4.
"""

import argparse
import importlib
import os
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from nets.omni.modules.sana_prompt_bridge import SanaPromptBridge
from nets.omni.modules.sana_prompt_bridge_gemma4 import Gemma4SanaPromptBridge
from nets.omni.modules.sana_prompt_bridge_qwen3vl import Qwen3VLSanaPromptBridge
from nets.third_party.sana.diffusion.longsana.utils.model_wrapper import SanaTextEncoder
from tools.inference.runtime_helpers import apply_lora_to_module, preprocess_prompts, to_attrdict
from diffusion.model.utils import prepare_prompt_ar
from diffusion.data.datasets import utils as sana_dataset_utils


def _load_sana_inference_backend(backend_name: str):
    if backend_name == "legacy":
        return importlib.import_module("tools.inference.sana_video_inference")
    if backend_name == "fixed":
        return importlib.import_module("tools.inference.sana_video_inference_fixed")
    raise ValueError(f"Unsupported SANA backend: {backend_name}")


def _get_base_ratios(config, height, width):
    image_size = getattr(getattr(config, "model", {}), "image_size", None) or height
    if getattr(config.vae, "vae_downsample_rate", 8) in [16, 32]:
        ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST_DIV32"
    else:
        ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST"
    base_ratios = getattr(sana_dataset_utils, ratio_name, None)
    if base_ratios is None:
        base_ratios = {f"{height / width:.2f}": [float(height), float(width)]}
    return base_ratios


def main():
    parser = argparse.ArgumentParser(description="Q1 student prompt embedding video test")
    parser.add_argument(
        "--sana-backend",
        type=str,
        default="fixed",
        choices=["legacy", "fixed"],
        help="Choose SANA inference backend module. Use 'fixed' unless you are intentionally debugging legacy sampling.",
    )
    parser.add_argument("--csv-path", type=str, default="data/openvid_q1/OpenVid_prompt_subset.csv")
    parser.add_argument("--prompt-index", type=int, default=0, help="Prompt index in CSV")
    parser.add_argument("--prompt", type=str, default=None, help="Direct prompt text (overrides CSV)")
    parser.add_argument(
        "--backbone-type",
        type=str,
        default=os.environ.get("STUDENT_BACKBONE_TYPE", "smolvlm2"),
        choices=["smolvlm2", "qwen3_vl", "qwen3vl", "qwen", "gemma4", "gemma_4"],
        help="Student text backbone type used by the bridge.",
    )
    parser.add_argument(
        "--smolvlm2-ckpt-path",
        type=str,
        default=os.environ.get("SMOLVLM2_CKPT_PATH", "omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"),
        help="Converted SmolVLM2 checkpoint used by the student bridge.",
    )
    parser.add_argument(
        "--tokenizer-model-id",
        type=str,
        default=os.environ.get("SMOLVLM2_TOKENIZER_MODEL_ID", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"),
        help="Tokenizer/model id for the student bridge text path.",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="omni_ckpts/sana_video_2b_480p")
    parser.add_argument("--config", type=str, default="configs/sana_video_config/Sana_2000M_480px_AdamW_fsdp.yaml")
    parser.add_argument("--bridge-ckpt", type=str, required=True, help="Q1 bridge checkpoint")
    parser.add_argument("--no-load-dit-from-ckpt", action="store_true", help="Do not load dit_trainable_state from bridge checkpoint")
    parser.add_argument("--adapter-ckpt-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="output/q1_student_video")
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument(
        "--latent-frames",
        type=int,
        default=None,
        help="Target latent temporal length T. If set, num_frames is auto-derived to match this T exactly.",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--negative-prompt", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--force-adapter-query-length", type=int, default=64)
    parser.add_argument("--override-gate", type=float, default=None, help="Override adapter_output_gate after loading checkpoint")
    parser.add_argument("--use-vision-head", action="store_true", help="Use vision head resampler before adapter")
    parser.add_argument("--adapter-in-channels", type=int, default=1024)
    parser.add_argument("--adapter-out-channels", type=int, default=2304)
    parser.add_argument("--adapter-enc-layers", type=int, default=2)
    parser.add_argument("--adapter-dec-layers", type=int, default=2)
    parser.add_argument("--adapter-ff-mult", type=int, default=2)
    parser.add_argument("--resampler-heads", type=int, default=8)
    parser.add_argument("--resampler-mlp-mult", type=int, default=2)
    parser.add_argument(
        "--projector-type",
        type=str,
        default="auto",
        choices=["auto", "legacy", "mcp_tiny", "mcp_full", "mcp_lexical_gated", "mcp_lexical_bottleneck"],
        help="Bridge projector type. 'auto' infers MCP usage from checkpoint.",
    )
    parser.add_argument("--mcp-hidden-dim", type=int, default=None)
    parser.add_argument("--mcp-num-fuse-layers", type=int, default=None)
    parser.add_argument("--mcp-use-refine", action="store_true")
    parser.add_argument("--mcp-refine-kernel-size", type=int, default=3)
    parser.add_argument("--mcp-fusion-temperature", type=float, default=1.0)
    parser.add_argument("--lora-enable", action="store_true")
    parser.add_argument("--no-auto-lora-from-ckpt", action="store_true")
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--sampling-algo", type=str, default=None)
    parser.add_argument("--motion-score", type=int, default=10)
    parser.add_argument(
        "--disable-motion-score",
        action="store_true",
        help="Do not append 'motion score: X.' to prompt before bridge encoding.",
    )
    parser.add_argument("--use-chi-prompt", action="store_true")
    parser.add_argument(
        "--no-bridge-post-layernorm",
        action="store_true",
        help="Disable post-bridge LayerNorm. By default inference applies LayerNorm to match training pipeline.",
    )
    parser.add_argument("--dit-lora-enable", action="store_true", help="Enable DiT LoRA modules before loading dit_trainable_state")
    parser.add_argument("--dit-lora-r", type=int, default=None)
    parser.add_argument("--dit-lora-alpha", type=int, default=None)
    parser.add_argument("--dit-lora-dropout", type=float, default=0.05)
    parser.add_argument("--flow-shift", type=float, default=None, help="Override inference flow shift")
    parser.add_argument(
        "--dual-text",
        action="store_true",
        help="Force dual-text inference: native SANA text as main branch, student bridge as auxiliary branch.",
    )
    args = parser.parse_args()
    default_num_frames = int(parser.get_default("num_frames"))

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    sana_backend = _load_sana_inference_backend(args.sana_backend)
    runtime_backend = sana_backend
    if args.sana_backend == "legacy":
        # Loader from old script is brittle with current registry/config layout.
        # Keep legacy sampler path, but use fixed backend for robust model bootstrap.
        runtime_backend = _load_sana_inference_backend("fixed")
        print("WARNING: using legacy sampling path for debugging only; current results may be unreliable.")
    print(f"SANA backend: {args.sana_backend}")
    print(
        "Infer request: backend=%s device=%s dtype=%s seed=%d steps=%d cfg_scale=%.4f "
        "sampling_algo_arg=%s flow_shift_arg=%s bridge_ckpt=%s"
        % (
            args.sana_backend,
            str(device),
            args.dtype,
            int(args.seed),
            int(args.steps),
            float(args.cfg_scale),
            (args.sampling_algo if args.sampling_algo is not None else "auto"),
            (str(args.flow_shift) if args.flow_shift is not None else "auto"),
            str(args.bridge_ckpt),
        )
    )

    if not os.path.exists(args.checkpoint_dir) and hasattr(runtime_backend, "download_checkpoint"):
        print(f"SANA checkpoint dir missing, auto-downloading to: {args.checkpoint_dir}")
        runtime_backend.download_checkpoint(local_dir=args.checkpoint_dir)

    prompt_arg = None if args.prompt is None else str(args.prompt)
    if prompt_arg == "__EMPTY__":
        prompt = ""
    elif prompt_arg is not None and prompt_arg.strip():
        prompt = prompt_arg.strip()
    else:
        df = pd.read_csv(args.csv_path)
        prompts = df["caption"].dropna().astype(str).tolist()
        prompt = prompts[args.prompt_index % len(prompts)]

    ckpt = torch.load(args.bridge_ckpt, map_location="cpu")
    infer_hints = ckpt.get("infer_hints", {}) if isinstance(ckpt, dict) else {}
    if not isinstance(infer_hints, dict):
        infer_hints = {}
    dit_state = ckpt.get("dit_trainable_state", {}) if isinstance(ckpt, dict) else {}

    config = runtime_backend.load_config_file(args.config)
    sampling_algo = args.sampling_algo or getattr(config.scheduler, "vis_sampler", "flow_euler")

    dual_text_enabled = bool(args.dual_text)
    if not dual_text_enabled and isinstance(dit_state, dict):
        dual_text_enabled = any(k.startswith("image_embedder.") for k in dit_state.keys())
    if dual_text_enabled:
        setattr(config.model, "cross_attn_image_embeds", True)
        setattr(
            config.model,
            "image_embed_channels",
            int(getattr(config.text_encoder, "caption_channels", 2304)),
        )
        print(
            "Dual-text inference enabled: native SANA main text + student auxiliary branch "
            f"(image_embed_channels={int(getattr(config.model, 'image_embed_channels', 2304))})"
        )

    # Load SANA model + VAE
    vae_dtype = torch.float32
    latent_size = args.height // config.vae.vae_downsample_rate
    models = runtime_backend.load_sana_models(
        config=config,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
        model_dtype=dtype,
        vae_dtype=vae_dtype,
        latent_size=latent_size,
        load_text_encoder=False,
    )

    # Prepare prompt canonical text (AR-normalized like SANA).
    base_ratios = _get_base_ratios(config, args.height, args.width)
    prompt_clean, _, hw, _, _ = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
    height, width = int(hw[0, 0].item()), int(hw[0, 1].item())
    prompt_plain = prompt_clean.strip()

    if args.sampling_algo is None:
        hint_sampler = str(infer_hints.get("train_vis_sampler", "")).strip()
        if hint_sampler:
            sampling_algo = hint_sampler

    # Build student prompt with train-time preprocessing hints when available.
    has_prompt_hints = any(
        k in infer_hints
        for k in (
            "train_use_chi_prompt",
            "train_use_prompt_templates",
            "train_prompt_templates",
            "train_motion_score",
        )
    )
    hint_motion_score = int(infer_hints.get("train_motion_score", args.motion_score))
    hint_templates = infer_hints.get("train_prompt_templates", [])
    if not isinstance(hint_templates, list):
        hint_templates = []
    hint_use_chi_prompt = bool(infer_hints.get("train_use_chi_prompt", False))
    hint_use_prompt_templates = bool(infer_hints.get("train_use_prompt_templates", False))
    hint_strict_sana_parity = bool(infer_hints.get("strict_sana_parity_text_path", False))
    hint_strict_sana_use_full_text_window = bool(infer_hints.get("strict_sana_use_full_text_window", False))
    hint_strict_sana_token_select_strategy = str(
        infer_hints.get("strict_sana_token_select_strategy", "tail") or "tail"
    )
    hint_strict_sana_head_tokens = int(infer_hints.get("strict_sana_head_tokens", 96) or 96)
    hint_strict_sana_tail_tokens = int(infer_hints.get("strict_sana_tail_tokens", 96) or 96)
    hint_strict_fail_fast_mask = bool(infer_hints.get("strict_fail_fast_mask", hint_strict_sana_parity))
    hint_sana_model_max_length = int(
        infer_hints.get("sana_model_max_length", getattr(config.text_encoder, "model_max_length", 300)) or 300
    )
    hint_student_max_length = int(infer_hints.get("train_student_max_length", 512) or 512)
    hint_student_pre_dit_layernorm = bool(infer_hints.get("mcp_pre_dit_layernorm", True))

    use_chi_prompt = bool(args.use_chi_prompt or hint_use_chi_prompt)
    if args.disable_motion_score:
        use_prompt_templates = False
    else:
        # For old checkpoints without prompt hints, keep legacy behavior:
        # append "motion score: X." in infer prompt path.
        use_prompt_templates = bool(hint_use_prompt_templates) if has_prompt_hints else True
    if use_prompt_templates and not hint_templates:
        hint_templates = ["{prompt} motion score: {motion_score}."]

    chi_prompt_text = ""
    if use_chi_prompt:
        chi_list = getattr(config.text_encoder, "chi_prompt", None)
        if isinstance(chi_list, (list, tuple)) and len(chi_list) > 0:
            chi_prompt_text = "\n".join(str(x) for x in chi_list)

    prep_cfg = to_attrdict(
        {
            "data": {
                "motion_score": int(hint_motion_score),
                "prompt_templates": hint_templates,
                "preprocessing": {
                    "normalize_whitespace": True,
                    "strip": True,
                    "remove_double_newlines": True,
                    "use_chi_prompt": use_chi_prompt,
                    "use_prompt_templates": use_prompt_templates,
                    "max_prompt_tokens": 200,
                },
            }
        }
    )
    student_prompt = preprocess_prompts(
        [prompt_plain],
        prep_cfg,
        random.Random(args.seed),
        tokenizer=None,
        chi_prompt=chi_prompt_text,
    )[0]
    print(
        "Prompt parity: use_chi_prompt=%s use_prompt_templates=%s motion_score=%d"
        % (use_chi_prompt, use_prompt_templates, int(hint_motion_score))
    )

    state = ckpt.get("student_state", ckpt)
    ckpt_has_lora = isinstance(state, dict) and ("smolvlm2_lora" in state)
    ckpt_has_projector = isinstance(state, dict) and ("projector" in state)
    # Priority for projector type:
    # CLI override > checkpoint infer_hints > heuristic from checkpoint keys.
    projector_type = args.projector_type
    if projector_type == "auto":
        hint_projector = str(infer_hints.get("projector_type", "")).strip()
        projector_type = hint_projector if hint_projector else ("mcp_full" if ckpt_has_projector else "legacy")

    # Same precedence for LoRA knobs: CLI > infer_hints > inferred from saved state.
    hint_student_lora_enable = bool(infer_hints.get("student_lora_enable", False))
    lora_enable = bool(
        args.lora_enable or hint_student_lora_enable or (ckpt_has_lora and (not args.no_auto_lora_from_ckpt))
    )

    def _infer_rank_from_lora_state(lora_state):
        if not isinstance(lora_state, dict):
            return None
        for name, tensor in lora_state.items():
            if name.endswith("lora_A") and hasattr(tensor, "shape") and len(tensor.shape) == 2:
                return int(tensor.shape[0])
        return None

    projector_state = state.get("projector", {}) if isinstance(state, dict) else {}
    inferred_lora_r = _infer_rank_from_lora_state(state.get("smolvlm2_lora", {})) if ckpt_has_lora else None
    hint_student_lora_r = int(infer_hints.get("student_lora_r", 0) or 0)
    hint_student_lora_alpha = int(infer_hints.get("student_lora_alpha", 0) or 0)
    lora_r = int(args.lora_r) if args.lora_r is not None else (hint_student_lora_r or inferred_lora_r or 8)
    lora_alpha = int(args.lora_alpha) if args.lora_alpha is not None else (hint_student_lora_alpha or (2 * lora_r))

    inferred_dit_lora_r = _infer_rank_from_lora_state(dit_state)
    hint_dit_lora_enable = bool(infer_hints.get("dit_lora_enable", False))
    hint_dit_lora_r = int(infer_hints.get("dit_lora_r", 0) or 0)
    hint_dit_lora_alpha = int(infer_hints.get("dit_lora_alpha", 0) or 0)
    dit_lora_enable = bool(args.dit_lora_enable or hint_dit_lora_enable or inferred_dit_lora_r is not None)
    dit_lora_r = int(args.dit_lora_r) if args.dit_lora_r is not None else (hint_dit_lora_r or inferred_dit_lora_r or 8)
    dit_lora_alpha = int(args.dit_lora_alpha) if args.dit_lora_alpha is not None else (hint_dit_lora_alpha or (2 * dit_lora_r))

    mcp_hidden_dim = int(args.mcp_hidden_dim) if args.mcp_hidden_dim is not None else None
    mcp_num_fuse_layers = int(args.mcp_num_fuse_layers) if args.mcp_num_fuse_layers is not None else None
    mcp_use_refine = bool(args.mcp_use_refine)
    mcp_refine_kernel_size = int(args.mcp_refine_kernel_size)
    if mcp_hidden_dim is None and infer_hints.get("mcp_hidden_dim", None) is not None:
        mcp_hidden_dim = int(infer_hints["mcp_hidden_dim"])
    if mcp_num_fuse_layers is None and infer_hints.get("mcp_num_fuse_layers", None) is not None:
        mcp_num_fuse_layers = int(infer_hints["mcp_num_fuse_layers"])
    if (not mcp_use_refine) and infer_hints.get("mcp_use_refine", None) is not None:
        mcp_use_refine = bool(infer_hints["mcp_use_refine"])
    if infer_hints.get("mcp_refine_kernel_size", None) is not None and args.mcp_refine_kernel_size == 3:
        mcp_refine_kernel_size = int(infer_hints["mcp_refine_kernel_size"])
    mcp_lexical_bottleneck_dim = int(infer_hints.get("mcp_lexical_bottleneck_dim", 256) or 256)
    mcp_lexical_gate_init = float(infer_hints.get("mcp_lexical_gate_init", 0.05) or 0.05)
    if projector_type in ("mcp_tiny", "mcp_full", "mcp_lexical_gated", "mcp_lexical_bottleneck") and isinstance(projector_state, dict) and projector_state:
        if mcp_hidden_dim is None and "compress.weight" in projector_state:
            mcp_hidden_dim = int(projector_state["compress.weight"].shape[0])
        if mcp_num_fuse_layers is None and "layer_w" in projector_state:
            mcp_num_fuse_layers = int(projector_state["layer_w"].shape[0])
        if (not mcp_use_refine) and any(k.startswith("refine.") for k in projector_state.keys()):
            mcp_use_refine = True
        if "refine.dw.weight" in projector_state and projector_state["refine.dw.weight"].ndim >= 3:
            mcp_refine_kernel_size = int(projector_state["refine.dw.weight"].shape[-1])
    if mcp_hidden_dim is None:
        mcp_hidden_dim = 512
    if mcp_num_fuse_layers is None:
        mcp_num_fuse_layers = 2

    # Student embeddings via bridge
    backbone_type = str(args.backbone_type or "smolvlm2").lower()
    if backbone_type in {"qwen3_vl", "qwen3vl", "qwen"}:
        bridge_cls = Qwen3VLSanaPromptBridge
    elif backbone_type in {"gemma4", "gemma_4"}:
        bridge_cls = Gemma4SanaPromptBridge
    else:
        bridge_cls = SanaPromptBridge
    bridge_kwargs = dict(
        adapter_ckpt_dir=args.adapter_ckpt_dir,
        adapter_in_channels=args.adapter_in_channels,
        adapter_out_channels=args.adapter_out_channels,
        adapter_query_length=64,
        adapter_num_encoder_layers=args.adapter_enc_layers,
        adapter_num_decoder_layers=args.adapter_dec_layers,
        adapter_ff_mult=args.adapter_ff_mult,
        num_prompt_queries=config.text_encoder.model_max_length,
        caption_channels=getattr(config.text_encoder, "caption_channels", 2304),
        precision_dtype=dtype,
        device=device,
        force_adapter_query_length=args.force_adapter_query_length,
        max_length=hint_student_max_length,
        use_vision_head=args.use_vision_head,
        resampler_num_heads=args.resampler_heads,
        resampler_mlp_mult=args.resampler_mlp_mult,
        projector_type=projector_type,
        mcp_hidden_dim=mcp_hidden_dim,
        mcp_num_fuse_layers=mcp_num_fuse_layers,
        mcp_use_refine=mcp_use_refine,
        mcp_refine_kernel_size=mcp_refine_kernel_size,
        mcp_fusion_temperature=args.mcp_fusion_temperature,
        mcp_lexical_bottleneck_dim=mcp_lexical_bottleneck_dim,
        mcp_lexical_gate_init=mcp_lexical_gate_init,
        strict_sana_parity_text_path=hint_strict_sana_parity,
        strict_sana_use_full_text_window=hint_strict_sana_use_full_text_window,
        strict_sana_token_select_strategy=hint_strict_sana_token_select_strategy,
        strict_sana_head_tokens=hint_strict_sana_head_tokens,
        strict_sana_tail_tokens=hint_strict_sana_tail_tokens,
        fail_fast_mask=hint_strict_fail_fast_mask,
        sana_model_max_length=hint_sana_model_max_length,
        sana_chi_prompt=chi_prompt_text,
        tokenizer_model_id=args.tokenizer_model_id,
    )
    if bridge_cls is SanaPromptBridge:
        bridge_kwargs.update(
            smolvlm2_ckpt_path=args.smolvlm2_ckpt_path,
            smol_vh_num_queries=1,
            lora_enable=lora_enable,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    elif bridge_cls is Qwen3VLSanaPromptBridge:
        bridge_kwargs.update(
            qwen_ckpt_path=args.smolvlm2_ckpt_path,
            gate_min_value=0.0,
            lora_enable=False,
        )
    else:
        bridge_kwargs.update(
            gemma_ckpt_path=args.smolvlm2_ckpt_path,
            gate_min_value=0.0,
            lora_enable=False,
        )
    bridge = bridge_cls(**bridge_kwargs)
    bridge_tokenizer = bridge._get_tokenizer()
    bridge_tokenizer_cls = type(bridge_tokenizer).__name__
    bridge_tokenizer_mod = type(bridge_tokenizer).__module__
    bridge_tokenizer_name = getattr(bridge_tokenizer, "name_or_path", None)
    print(
        "Bridge tokenizer: class=%s module=%s name_or_path=%s"
        % (bridge_tokenizer_cls, bridge_tokenizer_mod, bridge_tokenizer_name)
    )
    if bridge_tokenizer_cls == "SimpleTokenizer":
        raise RuntimeError(
            "SimpleTokenizer fallback detected in inference. "
            "Please ensure HuggingFace tokenizer cache is available for SmolVLM2 tokenizer_model_id."
        )

    print(
        f"Inference settings: projector_type={projector_type} "
        f"mcp_hidden_dim={mcp_hidden_dim} mcp_num_fuse_layers={mcp_num_fuse_layers} "
        f"mcp_use_refine={mcp_use_refine} mcp_refine_kernel_size={mcp_refine_kernel_size} "
        f"strict_sana_parity={hint_strict_sana_parity} fail_fast_mask={hint_strict_fail_fast_mask} "
        f"sana_model_max_length={hint_sana_model_max_length} student_max_length={hint_student_max_length} "
        f"student_lora_enable={lora_enable} r={lora_r} alpha={lora_alpha} "
        f"dit_lora_enable={dit_lora_enable} dit_r={dit_lora_r} dit_alpha={dit_lora_alpha}"
    )
    if infer_hints:
        print(f"Loaded infer_hints from checkpoint: {infer_hints}")

    if dit_lora_enable:
        replaced = apply_lora_to_module(
            models["diffusion_model"],
            target_modules=["q_linear", "kv_linear", "proj"],
            r=int(dit_lora_r),
            alpha=int(dit_lora_alpha),
            dropout=float(args.dit_lora_dropout),
            include_patterns=["cross_attn"],
            exclude_patterns=[],
        )
        print(f"Applied DiT LoRA modules for inference: replaced_linear_layers={replaced}")

    if (not args.no_load_dit_from_ckpt) and isinstance(ckpt, dict) and ("dit_trainable_state" in ckpt):
        missing, unexpected = models["diffusion_model"].load_state_dict(dit_state, strict=False)
        loaded = max(0, len(dit_state) - len(unexpected))
        print(
            f"Loaded dit_trainable_state from checkpoint: "
            f"keys={len(dit_state)}, loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if len(dit_state) > 0 and loaded == 0:
            print("WARNING: No DiT trainable keys were loaded. Check LoRA injection / target modules.")
    if "smolvlm2_vision_head" in state and getattr(bridge, "smolvlm2_vision_head", None) is not None:
        bridge.smolvlm2_vision_head.load_state_dict(state["smolvlm2_vision_head"])
    if "smolvlm2_text_trainable" in state:
        named = dict(bridge.smolvlm2_model.named_parameters())
        loaded = 0
        for name, tensor in state["smolvlm2_text_trainable"].items():
            target = named.get(name)
            if target is None:
                continue
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded += 1
        print(f"Loaded trainable SmolVLM2 params from checkpoint: {loaded}")
    if "smolvlm2_lora" in state:
        named = dict(bridge.smolvlm2_model.named_parameters())
        loaded = 0
        for name, tensor in state["smolvlm2_lora"].items():
            target = named.get(name)
            if target is None:
                continue
            target.data.copy_(tensor.to(device=target.device, dtype=target.dtype))
            loaded += 1
        print(f"Loaded LoRA params from checkpoint: {loaded}")
    if "adapter" in state and hasattr(bridge, "adapter"):
        try:
            bridge.adapter.load_state_dict(state["adapter"], strict=False)
        except Exception as e:
            print(f"Skip adapter load: {e}")
    if "adapter_output_norm" in state and hasattr(bridge, "adapter_output_norm"):
        try:
            bridge.adapter_output_norm.load_state_dict(state["adapter_output_norm"], strict=False)
        except Exception as e:
            print(f"Skip adapter_output_norm load: {e}")
    if "adapter_output_gate" in state and hasattr(bridge, "adapter_output_gate"):
        try:
            bridge.adapter_output_gate.data.copy_(state["adapter_output_gate"])
        except Exception as e:
            print(f"Skip adapter_output_gate load: {e}")
    if "resampler" in state and hasattr(bridge, "resampler"):
        try:
            bridge.resampler.load_state_dict(state["resampler"], strict=False)
        except Exception as e:
            print(f"Skip resampler load: {e}")
    if "projector" in state and getattr(bridge, "projector", None) is not None:
        missing, unexpected = bridge.projector.load_state_dict(state["projector"], strict=False)
        print(f"Loaded projector from checkpoint: missing={len(missing)} unexpected={len(unexpected)}")
    if args.override_gate is not None:
        bridge.adapter_output_gate.data.fill_(float(args.override_gate))
    print(f"Bridge gate value: {float(bridge.adapter_output_gate.detach().float().item()):.8f}")

    native_use_chi_prompt = bool(infer_hints.get("train_use_chi_prompt", False))
    native_text_encoder = None
    if dual_text_enabled:
        native_text_encoder = SanaTextEncoder(config, device=device, dtype=dtype)

    # Build conditional/unconditional embeddings.
    with torch.no_grad():
        aux_embeddings, aux_mask = bridge([student_prompt], return_mask=True)  # [B, L, C], [B, L]
        negative_aux_embeddings = None
        negative_aux_mask = None
        text_embeddings = aux_embeddings
        text_mask = aux_mask
        negative_embeddings = None
        neg_mask = None
        if args.cfg_scale > 1.0:
            negative_aux_embeddings, negative_aux_mask = bridge([args.negative_prompt], return_mask=True)
        # Match training path:
        # train_stage1_teacher_free.py may apply an extra LayerNorm on bridge outputs
        # before the DiT. Respect the train-time infer_hints by default.
        apply_bridge_post_layernorm = hint_student_pre_dit_layernorm and (not args.no_bridge_post_layernorm)
        if apply_bridge_post_layernorm:
            aux_embeddings = F.layer_norm(aux_embeddings, (aux_embeddings.shape[-1],))
            if negative_aux_embeddings is not None:
                negative_aux_embeddings = F.layer_norm(negative_aux_embeddings, (negative_aux_embeddings.shape[-1],))
        if dual_text_enabled:
            native_out = native_text_encoder.forward_chi([prompt_plain], use_chi_prompt=native_use_chi_prompt)
            text_embeddings = native_out["prompt_embeds"].to(device=device, dtype=dtype)
            text_mask = native_out["mask"].to(device=device, dtype=torch.long)
            if args.cfg_scale > 1.0:
                native_neg_out = native_text_encoder.forward_chi(
                    [args.negative_prompt],
                    use_chi_prompt=native_use_chi_prompt,
                )
                negative_embeddings = native_neg_out["prompt_embeds"].to(device=device, dtype=dtype)
                neg_mask = native_neg_out["mask"].to(device=device, dtype=torch.long)
        else:
            text_embeddings = aux_embeddings
            text_mask = aux_mask
            negative_embeddings = negative_aux_embeddings
            neg_mask = negative_aux_mask

    def _pad_or_truncate_seq(x: torch.Tensor, target_len: int) -> torch.Tensor:
        # x: [B, L, C]
        cur_len = int(x.shape[1])
        if cur_len == target_len:
            return x
        if cur_len > target_len:
            return x[:, :target_len, :]
        pad_len = target_len - cur_len
        pad = torch.zeros((x.shape[0], pad_len, x.shape[2]), device=x.device, dtype=x.dtype)
        return torch.cat([x, pad], dim=1)

    def _pad_or_truncate_mask(mask: torch.Tensor, target_len: int) -> torch.Tensor:
        # mask: [B, L]
        cur_len = int(mask.shape[1])
        if cur_len == target_len:
            return mask
        if cur_len > target_len:
            return mask[:, :target_len]
        pad_len = target_len - cur_len
        pad = torch.zeros((mask.shape[0], pad_len), device=mask.device, dtype=mask.dtype)
        return torch.cat([mask, pad], dim=1)

    # For CFG, cond/uncond token lengths can differ.
    # Normalize both sides to same sequence length so cross-attn mask stacking is valid.
    if negative_embeddings is not None:
        cond_len = int(text_embeddings.shape[1])
        uncond_len = int(negative_embeddings.shape[1])
        target_len = max(cond_len, uncond_len)
        text_embeddings = _pad_or_truncate_seq(text_embeddings, target_len)
        negative_embeddings = _pad_or_truncate_seq(negative_embeddings, target_len)
        if text_mask is None:
            text_mask = torch.ones((1, cond_len), device=device, dtype=torch.long)
        if neg_mask is None:
            neg_mask = torch.ones((1, uncond_len), device=device, dtype=torch.long)
        text_mask = _pad_or_truncate_mask(text_mask.to(device=device, dtype=torch.long), target_len)
        neg_mask = _pad_or_truncate_mask(neg_mask.to(device=device, dtype=torch.long), target_len)
    if negative_aux_embeddings is not None:
        cond_aux_len = int(aux_embeddings.shape[1])
        uncond_aux_len = int(negative_aux_embeddings.shape[1])
        target_aux_len = max(cond_aux_len, uncond_aux_len)
        aux_embeddings = _pad_or_truncate_seq(aux_embeddings, target_aux_len)
        negative_aux_embeddings = _pad_or_truncate_seq(negative_aux_embeddings, target_aux_len)
        if aux_mask is None:
            aux_mask = torch.ones((1, cond_aux_len), device=device, dtype=torch.long)
        else:
            aux_mask = aux_mask.to(device=device, dtype=torch.long)
        if negative_aux_mask is None:
            negative_aux_mask = torch.ones((1, uncond_aux_len), device=device, dtype=torch.long)
        else:
            negative_aux_mask = negative_aux_mask.to(device=device, dtype=torch.long)
        aux_mask = _pad_or_truncate_mask(aux_mask, target_aux_len)
        negative_aux_mask = _pad_or_truncate_mask(negative_aux_mask, target_aux_len)

    # Match SANA expected shape: [B, 1, L, C]
    text_embeddings = text_embeddings.unsqueeze(1)
    if negative_embeddings is not None:
        negative_embeddings = negative_embeddings.unsqueeze(1)

    # Mask must match bridge token length (L) for cross-attention.
    emb_masks = text_mask
    if emb_masks is None:
        emb_masks = torch.ones((1, text_embeddings.shape[2]), device=device, dtype=torch.long)
    else:
        emb_masks = emb_masks.to(device=device, dtype=torch.long)
    if negative_embeddings is not None:
        if neg_mask is None:
            neg_mask = torch.ones((1, text_embeddings.shape[2]), device=device, dtype=torch.long)
        else:
            neg_mask = neg_mask.to(device=device, dtype=torch.long)
        emb_masks = torch.cat([neg_mask, emb_masks], dim=0)

    # Prepare latents
    vae_latent_dim = config.vae.vae_latent_dim
    vae_downsample_rate = config.vae.vae_downsample_rate
    vae_stride = getattr(config.vae, "vae_stride", [1, vae_downsample_rate, vae_downsample_rate])
    vae_stride_t = vae_stride[0] if isinstance(vae_stride, list) and len(vae_stride) >= 1 else 1
    latent_h = height // vae_downsample_rate
    latent_w = width // vae_downsample_rate
    num_frames = int(args.num_frames)
    target_latent_t = int(args.latent_frames) if args.latent_frames is not None else None
    if target_latent_t is None and num_frames == default_num_frames:
        # Prefer raw/train latent T for generation-time temporal setup.
        # train_effective_latent_t may reflect train-only latent windowing.
        hinted_effective_t = infer_hints.get("train_effective_latent_t", None)
        hinted_train_t = infer_hints.get("train_latent_t", None)
        hinted_expected_t = infer_hints.get("train_expected_latent_t", None)
        hinted_chunk_index = infer_hints.get("train_chunk_index", None)
        hinted_use_process = bool(infer_hints.get("train_use_process_timesteps", False))
        hinted_frame_num = infer_hints.get("train_frame_num", None)
        if hinted_frame_num is None:
            hinted_frame_num = infer_hints.get("train_expected_frame_num", None)
        try:
            _train_t_int = int(hinted_train_t) if hinted_train_t is not None else None
            _eff_t_int = int(hinted_effective_t) if hinted_effective_t is not None else None
            if (
                _train_t_int is not None
                and _eff_t_int is not None
                and _train_t_int != _eff_t_int
                and hinted_chunk_index in (None, "None")
                and not hinted_use_process
            ):
                print(
                    "WARNING: checkpoint reports train_latent_t="
                    f"{_train_t_int} but train_effective_latent_t={_eff_t_int} "
                    "(train-time latent windowing). "
                    f"Use --latent_frames {_eff_t_int} for strict parity with this checkpoint."
                )
        except Exception:
            pass
        hinted_auto_t = hinted_train_t
        hinted_auto_t_src = "train_latent_t"
        if hinted_auto_t is None:
            hinted_auto_t = hinted_expected_t
            hinted_auto_t_src = "train_expected_latent_t"
        if hinted_auto_t is None:
            hinted_auto_t = hinted_effective_t
            hinted_auto_t_src = "train_effective_latent_t"
        if hinted_auto_t is not None:
            try:
                target_latent_t = max(1, int(hinted_auto_t))
                print(
                    "Temporal auto-match from checkpoint: "
                    f"{hinted_auto_t_src}={target_latent_t}"
                )
            except Exception:
                target_latent_t = None
        elif hinted_frame_num is not None:
            try:
                num_frames = max(1, int(hinted_frame_num))
                print(
                    "Temporal auto-match from checkpoint: "
                    f"train_frame_num={num_frames}"
                )
            except Exception:
                num_frames = int(args.num_frames)
    if target_latent_t is not None:
        if target_latent_t < 1:
            raise ValueError(f"--latent-frames must be >= 1, got {target_latent_t}")
        # Inverse of latent_t = floor((num_frames - 1) / stride_t) + 1.
        # Choose exact frame count for a requested latent T.
        num_frames = (target_latent_t - 1) * int(vae_stride_t) + 1
        print(
            f"Temporal override: latent_t={target_latent_t}, vae_stride_t={vae_stride_t} "
            f"-> num_frames={num_frames}"
        )

    latent_size_t = int(num_frames - 1) // vae_stride_t + 1
    print(
        f"Temporal setup: num_frames={num_frames}, vae_stride_t={vae_stride_t}, latent_t={latent_size_t}"
    )
    if target_latent_t is not None and latent_size_t != target_latent_t:
        raise RuntimeError(
            f"Requested latent_t={target_latent_t} but got latent_t={latent_size_t}. "
            f"Check vae_stride_t={vae_stride_t}."
        )
    latent_shape = (1, vae_latent_dim, latent_size_t, latent_h, latent_w)

    # Match training path: data_info uses latent-space H/W, not pixel-space H/W.
    hw_tensor = torch.tensor([[latent_h, latent_w]], dtype=torch.float, device=device)
    emb_masks_4d = emb_masks.unsqueeze(1).unsqueeze(1) if emb_masks is not None and emb_masks.dim() == 2 else emb_masks
    model_kwargs = {"data_info": {"img_hw": hw_tensor}, "mask": emb_masks_4d}
    if dual_text_enabled:
        image_embeds = aux_embeddings
        image_embeds_mask = aux_mask
        if negative_aux_embeddings is not None:
            image_embeds = torch.cat([negative_aux_embeddings, image_embeds], dim=0)
            if image_embeds_mask is None:
                image_embeds_mask = torch.ones(
                    (1, aux_embeddings.shape[1]), device=device, dtype=torch.long
                )
            if negative_aux_mask is None:
                negative_aux_mask = torch.ones(
                    (1, negative_aux_embeddings.shape[1]), device=device, dtype=torch.long
                )
            image_embeds_mask = torch.cat(
                [
                    negative_aux_mask.to(device=device, dtype=torch.long),
                    image_embeds_mask.to(device=device, dtype=torch.long),
                ],
                dim=0,
            )
        elif image_embeds_mask is not None:
            image_embeds_mask = image_embeds_mask.to(device=device, dtype=torch.long)
        model_kwargs["data_info"]["image_embeds"] = image_embeds.to(device=device, dtype=dtype)
        if image_embeds_mask is not None:
            model_kwargs["data_info"]["image_embeds_mask"] = image_embeds_mask

    def _normalize_chunk_index(raw_chunk):
        if raw_chunk is None:
            return None
        if isinstance(raw_chunk, (int, float)):
            vals = [0, int(raw_chunk)]
        elif isinstance(raw_chunk, str):
            text = raw_chunk.strip()
            if text.startswith("[") and text.endswith("]"):
                text = text[1:-1]
            vals = [int(v.strip()) for v in text.split(",") if v.strip()]
        else:
            vals = [int(v) for v in list(raw_chunk)]
        if len(vals) == 1:
            vals = [0, int(vals[0])]
        vals = sorted(set(int(v) for v in vals))
        if len(vals) > 0 and vals[0] != 0:
            vals = [0] + vals
        return vals if vals else None

    train_chunk_index = infer_hints.get("train_chunk_index", None)
    if train_chunk_index is None:
        train_chunk_index = getattr(getattr(config, "model", {}), "chunk_index", None)
    train_chunk_index = _normalize_chunk_index(train_chunk_index)
    if train_chunk_index is not None:
        model_kwargs["chunk_index"] = train_chunk_index
        print(f"Using chunk_index in inference: {train_chunk_index}")

    # Sampling: prefer inference_flow_shift for generation parity with SANA inference.
    # Fallback order: CLI override -> checkpoint/config inference shift -> train shift.
    if args.flow_shift is not None:
        infer_flow_shift = float(args.flow_shift)
    else:
        infer_flow_shift = infer_hints.get("inference_flow_shift", None)
        if infer_flow_shift is None:
            infer_flow_shift = getattr(config.scheduler, "inference_flow_shift", None)
        if infer_flow_shift is None:
            infer_flow_shift = infer_hints.get("train_flow_shift", None)
        if infer_flow_shift is None:
            infer_flow_shift = getattr(config.scheduler, "flow_shift", 7.0)
        infer_flow_shift = float(infer_flow_shift)
    train_flow_shift = infer_hints.get("train_flow_shift", None)
    if train_flow_shift is not None:
        print(f"Flow shift: train={float(train_flow_shift):.4f}, infer={infer_flow_shift:.4f}")
    else:
        print(f"Flow shift: infer={infer_flow_shift:.4f}")
    print(
        "Sampling parameters: backend=%s seed=%d steps=%d cfg_scale=%.4f flow_shift=%.4f "
        "sampling_algo=%s num_frames=%d latent_shape=%s height=%d width=%d "
        "negative_prompt_len=%d motion_score=%d use_chi_prompt=%s prompt_index=%d dual_text=%s"
        % (
            args.sana_backend,
            int(args.seed),
            int(args.steps),
            float(args.cfg_scale),
            float(infer_flow_shift),
            str(sampling_algo),
            int(num_frames),
            tuple(int(v) for v in latent_shape),
            int(height),
            int(width),
            len(args.negative_prompt or ""),
            int(args.motion_score),
            bool(use_chi_prompt),
            int(args.prompt_index),
            bool(dual_text_enabled),
        )
    )
    print(f"Prompt text: {prompt_plain}")

    if args.sana_backend == "legacy":
        # Legacy path mirrors old sana_video_inference.py behavior.
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        with torch.no_grad():
            latents = sana_backend.simple_flow_matching_sampling(
                models["diffusion_model"],
                text_embeddings,
                latent_shape,
                num_steps=args.steps,
                device=str(device),
                dtype=dtype,
                cfg_scale=args.cfg_scale,
                flow_shift=infer_flow_shift,
            )
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
        latents = sana_backend.flow_matching_sampling(
            models["diffusion_model"],
            latents,
            text_embeddings,
            negative_embeddings,
            num_steps=args.steps,
            device=str(device),
            cfg_scale=args.cfg_scale,
            flow_shift=infer_flow_shift,
            model_kwargs=model_kwargs,
            sampling_algo=sampling_algo,
        )

    # Decode
    vae_type = config.vae.vae_type
    if args.sana_backend == "legacy":
        latents = latents.to(models.get("vae_dtype", latents.dtype))
        with torch.no_grad():
            decoded = runtime_backend.vae_decode(vae_type, models["vae"], latents)
        if isinstance(decoded, list):
            decoded = torch.stack(decoded, dim=0)
        if decoded.ndim == 5:
            # [B, C, T, H, W] -> [C, T, H, W]
            decoded = decoded[0]
        if decoded.ndim == 4 and decoded.shape[0] in (1, 3):
            # [C, T, H, W] -> [T, H, W, C]
            video = decoded.permute(1, 2, 3, 0).cpu().numpy()
        elif decoded.ndim == 4 and decoded.shape[1] in (1, 3):
            # [T, C, H, W] -> [T, H, W, C]
            video = decoded.permute(0, 2, 3, 1).cpu().numpy()
        else:
            raise RuntimeError(f"Unexpected legacy decode output shape: {tuple(decoded.shape)}")
        video = (video + 1.0) / 2.0
        video = np.clip(video, 0, 1)
        video = (video * 255).astype(np.uint8)
    else:
        latents = latents.to(models.get("vae_dtype", latents.dtype))
        video = models["vae"].decode(latents) if hasattr(models["vae"], "decode") else None
        if video is None:
            from diffusion.model.builder import vae_decode

            video = vae_decode(vae_type, models["vae"], latents)
        if isinstance(video, list):
            video = torch.stack(video, dim=0)
        video = video[0].permute(1, 2, 3, 0).cpu().numpy()
        video = (video + 1.0) / 2.0
        video = np.clip(video, 0, 1)
        video = (video * 255).astype(np.uint8)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_prompt = "".join(c for c in prompt_clean if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
    if video.shape[0] == 1:
        try:
            from PIL import Image

            png_path = os.path.join(args.output_dir, f"q1_student_{timestamp}_{safe_prompt[:40]}.png")
            Image.fromarray(video[0]).save(png_path)
            print(f"Saved image to: {png_path}")
        except Exception as e:
            print(f"Skip PNG export for single-frame output: {e}")
    else:
        out_path = os.path.join(args.output_dir, f"q1_student_{timestamp}_{safe_prompt[:40]}.mp4")
        runtime_backend.save_video(video, out_path, fps=16)
        print(f"Saved video to: {out_path}")


if __name__ == "__main__":
    main()
