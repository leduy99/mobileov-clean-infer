#!/usr/bin/env python3
"""
SANA-video inference script - rewritten following SANA repo structure and patterns.

This script follows the proper SANA inference pipeline:
- Uses config to load models
- Properly formats inputs for SANA model (x: [B,C,T,H,W], timestep, y: text embeddings)
- Implements flow matching sampling with CFG support
- Uses VAE decode properly

Usage:
    python sana_video_inference_fixed.py \
        --prompt "a cat playing with a wool beside the fireside" \
        --output_dir output/sana_inference \
        --checkpoint_dir omni_ckpts/sana_video_2b_480p
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
_sana_repo_root = os.path.join(project_root, "nets", "third_party", "sana")
if os.path.isdir(_sana_repo_root):
    sys.path.insert(0, _sana_repo_root)

import argparse
import logging
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
import shutil

warnings.filterwarnings('ignore')
os.environ.setdefault("DISABLE_XFORMERS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import yaml
from easydict import EasyDict
from tqdm import tqdm
import math

try:
    from diffusion import FlowEuler, DPMS
    SANA_FLOW_AVAILABLE = True
    SANA_DPM_AVAILABLE = True
except ImportError:
    FlowEuler = None
    DPMS = None
    SANA_FLOW_AVAILABLE = False
    SANA_DPM_AVAILABLE = False

try:
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
    DIFFUSERS_AVAILABLE = True
except ImportError:
    print("Warning: diffusers not available. Flow matching fallback will be disabled.")
    DIFFUSERS_AVAILABLE = False
    FlowMatchEulerDiscreteScheduler = None
    retrieve_timesteps = None

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV (cv2) not available. Video processing will be disabled.")
    CV2_AVAILABLE = False
    cv2 = None

# SANA imports
from diffusion.model.builder import (
    build_model,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_encode,
    vae_decode,
)
from diffusion.utils.config import SanaVideoConfig
from diffusion.model.utils import get_weight_dtype
from diffusion.data.datasets import utils as sana_dataset_utils
from diffusion.model.utils import prepare_prompt_ar


def init_logging():
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def set_env(seed=0, latent_size=256):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)
    for _ in range(30):
        torch.randn(1, 4, latent_size, latent_size)


def get_base_ratios(config, height, width):
    image_size = getattr(getattr(config, "model", {}), "image_size", None) or height
    if getattr(config.vae, "vae_downsample_rate", 8) in [16, 32]:
        ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST_DIV32"
    else:
        ratio_name = f"ASPECT_RATIO_VIDEO_{image_size}_TEST"
    base_ratios = getattr(sana_dataset_utils, ratio_name, None)
    if base_ratios is None:
        base_ratios = {f"{height/width:.2f}": [float(height), float(width)]}
    return base_ratios


def download_checkpoint(hf_model_id="Efficient-Large-Model/SANA-Video_2B_480p", 
                       local_dir="omni_ckpts/sana_video_2b_480p"):
    """Download checkpoint from HuggingFace."""
    local_dir = os.path.abspath(local_dir)
    
    if os.path.exists(local_dir):
        print(f"Checkpoint directory already exists: {local_dir}")
        return local_dir
    
    print(f"Downloading checkpoint from HuggingFace: {hf_model_id}")
    print(f"Target directory: {local_dir}")
    
    os.makedirs(local_dir, exist_ok=True)
    
    try:
        cmd = [
            "huggingface-cli", "download",
            hf_model_id,
            "--local-dir", local_dir,
            "--local-dir-use-symlinks", "False"
        ]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Download completed successfully!")
        return local_dir
    except subprocess.CalledProcessError as e:
        print(f"Error downloading checkpoint: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        print(f"\nPlease manually download from: https://huggingface.co/{hf_model_id}")
        raise
    except FileNotFoundError:
        print("Error: huggingface-cli not found!")
        print("Please install it with: pip install huggingface_hub[cli]")
        print(f"Or manually download from: https://huggingface.co/{hf_model_id}")
        raise


def load_config_file(config_path="configs/sana_video_2000M_480px.yaml"):
    """Load SANA-video config file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    try:
        import pyrallis
        with open(config_path, 'r') as f:
            config = pyrallis.load(SanaVideoConfig, f)
        return config
    except Exception as e:
        print(f"Warning: pyrallis load failed ({e}), falling back to yaml + EasyDict.")
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        # Convert to EasyDict for easy access
        config = EasyDict(config_dict)
        return config


def load_sana_models(
    config,
    checkpoint_dir="omni_ckpts/sana_video_2b_480p",
    device="cuda:0",
    model_dtype=torch.bfloat16,
    vae_dtype=torch.float32,
    latent_size=32,
    load_text_encoder=True,
):
    """Load SANA-video model, VAE, and text encoder following SANA patterns."""
    print("=" * 80)
    print("Loading SANA-video models...")
    print("=" * 80)
    
    # 1. Load text encoder (optional for student-bridge inference path)
    tokenizer = None
    text_encoder = None
    if load_text_encoder:
        print("\n[1/3] Loading text encoder...")
        text_encoder_name = config.text_encoder.text_encoder_name
        tokenizer, text_encoder = get_tokenizer_and_text_encoder(
            name=text_encoder_name,
            device=device
        )
        text_encoder.eval()
        print(f"✅ Text encoder loaded: {text_encoder_name}")
    else:
        print("\n[1/3] Skipping text encoder load (student bridge path)")
    
    # 2. Load VAE
    print("\n[2/3] Loading VAE...")
    vae_type = config.vae.vae_type
    vae_pretrained = config.vae.vae_pretrained
    
    # Handle HuggingFace path
    if vae_pretrained.startswith("hf://"):
        vae_path = vae_pretrained.replace("hf://", "")
        vae_path = vae_path.replace("Efficient-Large-Model/SANA-Video_2B_480p/", f"{checkpoint_dir}/")
    else:
        vae_path = vae_pretrained
    
    # Make path absolute
    if not os.path.isabs(vae_path):
        vae_path = os.path.abspath(vae_path)
    
    vae = get_vae(
        name=vae_type,
        model_path=vae_path,
        device=device,
        dtype=vae_dtype,
        config=config.vae
    )
    if hasattr(vae, 'eval'):
        vae.eval()
    print(f"✅ VAE loaded: {vae_type} from {vae_path}")
    
    # 3. Load diffusion model
    print("\n[3/3] Loading diffusion model...")
    model_type = config.model.model
    
    # Handle checkpoint path
    load_from = config.model.load_from
    if load_from.startswith("hf://"):
        checkpoint_path = load_from.replace("hf://", "")
        checkpoint_path = checkpoint_path.replace("Efficient-Large-Model/SANA-Video_2B_480p/", f"{checkpoint_dir}/")
    else:
        checkpoint_path = load_from
    
    # Make path absolute
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.abspath(checkpoint_path)
    
    # Build model config
    from diffusion.utils.config import model_video_init_config
    model_cfg = model_video_init_config(config, latent_size=latent_size)
    model_cfg['type'] = model_type
    
    # Build model
    diffusion_model = build_model(
        model_cfg,
        use_fp32_attention=getattr(config.model, "fp32_attention", False),
    )
    diffusion_model.eval()
    diffusion_model = diffusion_model.to(device).to(model_dtype)
    
    # Load checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remove prefix if exists
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Filter out pos_embed if shape mismatch (can be recomputed)
        model_pos_embed_shape = None
        if 'pos_embed' in diffusion_model.state_dict():
            model_pos_embed_shape = diffusion_model.state_dict()['pos_embed'].shape
        
        if 'pos_embed' in state_dict:
            ckpt_pos_embed_shape = state_dict['pos_embed'].shape
            if model_pos_embed_shape is not None and ckpt_pos_embed_shape != model_pos_embed_shape:
                print(f"Warning: pos_embed shape mismatch (checkpoint: {ckpt_pos_embed_shape}, model: {model_pos_embed_shape})")
                print("Removing pos_embed from checkpoint (will be recomputed)...")
                state_dict.pop('pos_embed')
        
        missing_keys, unexpected_keys = diffusion_model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
        
        print(f"✅ Diffusion model loaded: {model_type}")
    else:
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("Will use randomly initialized weights (not recommended for inference)")
    
    return {
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'vae_dtype': vae_dtype,
        'diffusion_model': diffusion_model,
        'config': config,
    }


@torch.no_grad()
def encode_text(tokenizer, text_encoder, prompt, config, device="cuda:0", use_chi_prompt=True):
    """Encode text prompt to embeddings following SANA format exactly."""
    if tokenizer is None:
        # For Qwen2.5-VL
        assert hasattr(text_encoder, 'encode_text')
        y = text_encoder.encode_text([prompt], device=device)
        return y, None, None

    model_max_length = getattr(config.text_encoder, "model_max_length", 300)
    chi_prompt_list = getattr(config.text_encoder, "chi_prompt", None) if use_chi_prompt else None
    if chi_prompt_list:
        chi_prompt = "\n".join(chi_prompt_list)
        prompts_all = [chi_prompt + prompt]
        num_chi_prompt_tokens = len(tokenizer.encode(chi_prompt))
        max_length_all = num_chi_prompt_tokens + model_max_length - 2
    else:
        prompts_all = [prompt]
        max_length_all = model_max_length

    tokens = tokenizer(
        prompts_all,
        max_length=max_length_all,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)

    text_embeddings = text_encoder(
        tokens.input_ids,
        attention_mask=tokens.attention_mask
    ).last_hidden_state  # [B, L, D]

    select_index = [0] + list(range(-model_max_length + 1, 0))
    text_embeddings = text_embeddings[:, None][:, :, select_index]  # [B, 1, model_max_length, D]
    emb_masks = tokens.attention_mask[:, select_index]  # [B, model_max_length]

    return text_embeddings, tokens, emb_masks


@torch.no_grad()
def encode_negative_prompt(tokenizer, text_encoder, negative_prompt, config, device="cuda:0"):
    """Encode negative prompt to embeddings following SANA format."""
    if tokenizer is None:
        assert hasattr(text_encoder, 'encode_text')
        y = text_encoder.encode_text([negative_prompt], device=device)
        return y, None

    model_max_length = getattr(config.text_encoder, "model_max_length", 300)
    tokens = tokenizer(
        negative_prompt,
        max_length=model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(device)
    text_embeddings = text_encoder(
        tokens.input_ids,
        attention_mask=tokens.attention_mask
    ).last_hidden_state  # [B, L, D]
    return text_embeddings[:, None], tokens.attention_mask


@torch.no_grad()
def flow_matching_sampling(
    model,
    latents,
    text_embeddings,
    negative_embeddings,
    num_steps=50,
    device="cuda:0",
    cfg_scale=7.0,
    flow_shift=7.0,
    model_kwargs=None,
    sampling_algo="flow_euler",
):
    """
    Flow matching sampling using SANA's FlowEuler when available.
    """
    print(f"Running flow matching sampling with {num_steps} steps, flow_shift={flow_shift}, cfg_scale={cfg_scale}...")

    if model_kwargs is None:
        model_kwargs = {}

    if sampling_algo in {"chunk_flow_euler", "flow_euler"}:
        sampling_algo = "flow_euler"
    elif sampling_algo == "flow_dpm-solver":
        pass
    else:
        raise ValueError(f"Unsupported sampling_algo={sampling_algo!r}. Expected flow_euler/chunk_flow_euler/flow_dpm-solver.")

    if sampling_algo == "flow_dpm-solver":
        if not SANA_DPM_AVAILABLE:
            raise RuntimeError("SANA DPMS sampler is not available; cannot run flow_dpm-solver.")
        dpm_solver = DPMS(
            model,
            condition=text_embeddings,
            uncondition=negative_embeddings,
            cfg_scale=cfg_scale,
            model_type="flow",
            guidance_type="classifier-free",
            model_kwargs=model_kwargs,
            schedule="FLOW",
        )
        return dpm_solver.sample(
            latents,
            steps=num_steps,
            order=2,
            skip_type="time_uniform_flow",
            method="multistep",
            flow_shift=flow_shift,
        )

    if SANA_FLOW_AVAILABLE:
        flow_solver = FlowEuler(
            model,
            condition=text_embeddings,
            uncondition=negative_embeddings,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            model_kwargs=model_kwargs,
        )
        return flow_solver.sample(latents, steps=num_steps)

    if not DIFFUSERS_AVAILABLE:
        raise RuntimeError("diffusers is not available; cannot run flow matching sampling.")

    scheduler = FlowMatchEulerDiscreteScheduler(shift=flow_shift)
    timesteps, _ = retrieve_timesteps(scheduler, num_steps, device, None)
    do_classifier_free_guidance = cfg_scale > 1.0

    if do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_embeddings, text_embeddings], dim=0)
    else:
        prompt_embeds = text_embeddings

    for t in tqdm(list(enumerate(timesteps)), desc="Sampling"):
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        timestep = t.expand(latent_model_input.shape[0])
        noise_pred = model(
            latent_model_input,
            timestep,
            prompt_embeds,
            **model_kwargs,
        )

        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

        latents_dtype = latents.dtype
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
            latents = latents.to(latents_dtype)

    return latents


@torch.no_grad()
def generate_video(
    models,
    prompt,
    num_frames=81,
    height=480,
    width=832,
    num_inference_steps=50,
    cfg_scale=7.0,
    seed=42,
    device="cuda:0",
    dtype=torch.bfloat16,
    sampling_algo="flow_euler",
    negative_prompt="",
    motion_score=10,
    high_motion=False,
    use_chi_prompt=True,
):
    """Generate video from text prompt following SANA patterns."""
    print("=" * 80)
    print(f"Generating video from prompt: {prompt}")
    print(f"Resolution: {width}x{height}, Frames: {num_frames}")
    print(f"Inference steps: {num_inference_steps}, CFG scale: {cfg_scale}")
    print(
        "Sampling parameters: backend=fixed seed=%d sampling_algo=%s num_steps=%d cfg_scale=%.4f "
        "num_frames=%d height=%d width=%d negative_prompt_len=%d motion_score=%d "
        "high_motion=%s use_chi_prompt=%s"
        % (
            int(seed),
            str(sampling_algo),
            int(num_inference_steps),
            float(cfg_scale),
            int(num_frames),
            int(height),
            int(width),
            len(negative_prompt or ""),
            int(motion_score),
            bool(high_motion),
            bool(use_chi_prompt),
        )
    )
    print("=" * 80)
    
    # Set seed
    generator = torch.Generator(device=device).manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    config = models['config']
    
    # Get flow shift from config
    flow_shift = getattr(config.scheduler, 'inference_flow_shift', None)
    if flow_shift is None:
        flow_shift = getattr(config.scheduler, 'flow_shift', 7.0)
    print(f"Using flow_shift={flow_shift} from config")
    
    # Encode text
    print("\n[1/4] Encoding text prompt...")
    if motion_score > 0:
        motion_prompt = f" motion score: {int(motion_score)}."
    else:
        motion_prompt = " high motion" if high_motion else " low motion"

    base_ratios = get_base_ratios(config, height, width)
    prompt_clean, _, hw, _, _ = prepare_prompt_ar(prompt, base_ratios, device=device, show=False)
    height, width = int(hw[0, 0].item()), int(hw[0, 1].item())
    prompt_with_motion = f"{prompt_clean.strip()}{motion_prompt}"

    text_embeddings, tokens, emb_masks = encode_text(
        models['tokenizer'],
        models['text_encoder'],
        prompt_with_motion,
        config,
        device=device,
        use_chi_prompt=use_chi_prompt,
    )

    negative_embeddings, negative_emb_masks = encode_negative_prompt(
        models['tokenizer'],
        models['text_encoder'],
        negative_prompt,
        config,
        device=device,
    )

    if cfg_scale > 1.0 and emb_masks is not None and negative_emb_masks is not None:
        emb_masks = torch.cat([negative_emb_masks, emb_masks], dim=0)

    emb_masks_shape = emb_masks.shape if emb_masks is not None else None
    print(f"✅ Text embeddings shape: {text_embeddings.shape}, emb_masks shape: {emb_masks_shape}")
    
    # Prepare latent shape
    # SANA calculates latent_size_t correctly: int(num_frames - 1) // vae_stride[0] + 1
    vae_latent_dim = config.vae.vae_latent_dim
    vae_downsample_rate = config.vae.vae_downsample_rate
    vae_stride = getattr(config.vae, 'vae_stride', [1, vae_downsample_rate, vae_downsample_rate])
    if isinstance(vae_stride, list) and len(vae_stride) >= 1:
        vae_stride_t = vae_stride[0]
    else:
        vae_stride_t = 1
    
    latent_h = height // vae_downsample_rate
    latent_w = width // vae_downsample_rate
    # Calculate latent_size_t like SANA gốc
    latent_size_t = int(num_frames - 1) // vae_stride_t + 1
    latent_shape = (1, vae_latent_dim, latent_size_t, latent_h, latent_w)
    print(f"\n[2/4] Latent shape: {latent_shape} (num_frames={num_frames}, vae_stride_t={vae_stride_t}, latent_size_t={latent_size_t})")
    
    # Sample from diffusion model
    print("\n[3/4] Running diffusion sampling...")

    # Prepare hw (height, width) like SANA gốc
    hw = torch.tensor([[height, width]], dtype=torch.float, device=device)
    
    model_kwargs = {
        'data_info': {'img_hw': hw},
    }
    if emb_masks is not None:
        model_kwargs['mask'] = emb_masks

    latents = torch.randn(latent_shape, device=device, dtype=dtype, generator=generator)
    latents = flow_matching_sampling(
        models['diffusion_model'],
        latents,
        text_embeddings,
        negative_embeddings,
        num_steps=num_inference_steps,
        device=device,
        cfg_scale=cfg_scale,
        flow_shift=flow_shift,
        model_kwargs=model_kwargs,
        sampling_algo=sampling_algo,
    )
    
    # Decode with VAE
    print("\n[4/4] Decoding latents to video...")
    vae_type = config.vae.vae_type
    latents = latents.to(models.get('vae_dtype', latents.dtype))
    
    video = vae_decode(vae_type, models['vae'], latents)
    if isinstance(video, list):
        video = torch.stack(video, dim=0)
    video = video[0]  # [C, T, H, W]
    
    # Convert to numpy: [C, T, H, W] -> [T, H, W, C]
    video = video.permute(1, 2, 3, 0).cpu().numpy()  # [T, H, W, C]
    
    # Normalize to [0, 255]
    video = (video + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    video = np.clip(video, 0, 1)
    video = (video * 255).astype(np.uint8)
    
    print(f"✅ Video generated: shape {video.shape}")
    
    return video


def save_video(video, output_path, fps=16):
    """Save video to file."""
    if not CV2_AVAILABLE:
        print("Warning: OpenCV not available, cannot save video")
        return False
    
    T, H, W, C = video.shape
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    
    # Write frames
    for frame in video:
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    
    writer.release()
    print(f"✅ Video saved to: {output_path}")
    return True


def extract_frames(video, output_dir, prefix="frame"):
    """Extract frames from video."""
    os.makedirs(output_dir, exist_ok=True)
    
    T = video.shape[0]
    for t in range(T):
        frame = video[t]
        frame_path = os.path.join(output_dir, f"{prefix}_{t:05d}.png")
        Image.fromarray(frame).save(frame_path)
    
    print(f"✅ Extracted {T} frames to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="SANA-video inference (following SANA repo patterns)")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
    parser.add_argument("--use_chi_prompt", action="store_true",
                       help="Force-enable CHI prompt prefix in text encoding")
    parser.add_argument("--disable_chi_prompt", action="store_true",
                       help="Force-disable CHI prompt prefix in text encoding")
    parser.add_argument("--output_dir", type=str, default="output/sana_inference",
                       help="Output directory")
    parser.add_argument("--config", type=str, default="configs/sana_video_2000M_480px.yaml",
                       help="Path to config file")
    parser.add_argument("--checkpoint_dir", type=str, default="omni_ckpts/sana_video_2b_480p",
                       help="Local checkpoint directory")
    parser.add_argument("--download_checkpoint", action="store_true",
                       help="Download checkpoint from HuggingFace")
    parser.add_argument("--num_frames", type=int, default=81,
                       help="Number of frames to generate")
    parser.add_argument("--height", type=int, default=480,
                       help="Video height")
    parser.add_argument("--width", type=int, default=832,
                       help="Video width")
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of inference steps")
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                       help="Classifier-free guidance scale")
    parser.add_argument("--sampling_algo", type=str, default=None,
                       help="Sampling algorithm: flow_euler or flow_dpm-solver")
    parser.add_argument("--negative_prompt", type=str, default=None,
                       help="Negative prompt for CFG (defaults to SANA preset)")
    parser.add_argument("--motion_score", type=int, default=10,
                       help="Motion score to append to prompt (<=0 to use high/low motion)")
    parser.add_argument("--high_motion", action="store_true",
                       help="Use high motion prompt when motion_score <= 0")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--extract_frames", action="store_true",
                       help="Extract frames from generated video")
    
    args = parser.parse_args()
    if args.use_chi_prompt and args.disable_chi_prompt:
        raise ValueError("Choose only one of --use_chi_prompt or --disable_chi_prompt")
    
    init_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download checkpoint if requested or missing locally.
    checkpoint_missing = not os.path.exists(args.checkpoint_dir)
    if args.download_checkpoint or checkpoint_missing:
        print("=" * 80)
        print("Downloading checkpoint...")
        print("=" * 80)
        download_checkpoint(local_dir=args.checkpoint_dir)
    
    # Load config
    config = load_config_file(args.config)
    
    # Update checkpoint paths in config if needed
    if config.model.load_from.startswith("hf://"):
        suffix = config.model.load_from.replace("hf://Efficient-Large-Model/SANA-Video_2B_480p/", "")
        config.model.load_from = os.path.join(args.checkpoint_dir, suffix)
    if config.vae.vae_pretrained.startswith("hf://"):
        suffix = config.vae.vae_pretrained.replace("hf://Efficient-Large-Model/SANA-Video_2B_480p/", "")
        config.vae.vae_pretrained = os.path.join(args.checkpoint_dir, suffix)
    
    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_dtype = get_weight_dtype(config.model.mixed_precision)
    vae_dtype = get_weight_dtype(config.vae.weight_dtype)

    # Match model input_size with resolution to avoid pos_embed mismatch.
    latent_size = args.height // config.vae.vae_downsample_rate

    set_env(args.seed, getattr(config.model, "image_size", args.height) // config.vae.vae_downsample_rate)
    
    models = load_sana_models(
        config,
        checkpoint_dir=args.checkpoint_dir,
        device=str(device),
        model_dtype=model_dtype,
        vae_dtype=vae_dtype,
        latent_size=latent_size,
    )

    sana_default_negative = (
        "A chaotic sequence with misshapen, deformed limbs in heavy motion blur, sudden disappearance, "
        "jump cuts, jerky movements, rapid shot changes, frames out of sync, inconsistent character shapes, "
        "temporal artifacts, jitter, and ghosting effects, creating a disorienting visual experience."
    )
    if args.negative_prompt is not None:
        negative_prompt = args.negative_prompt
    else:
        negative_prompt = getattr(config, "negative_prompt", "") or sana_default_negative

    sampling_algo = args.sampling_algo or getattr(config.scheduler, "vis_sampler", "flow_euler")
    if args.use_chi_prompt:
        use_chi_prompt = True
    elif args.disable_chi_prompt:
        use_chi_prompt = False
    else:
        use_chi_prompt = bool(getattr(config.text_encoder, "chi_prompt", None))
    print(f"Using CHI prompt prefix: {use_chi_prompt}")

    # Generate video
    video = generate_video(
        models,
        args.prompt,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        seed=args.seed,
        device=str(device),
        dtype=model_dtype,
        sampling_algo=sampling_algo,
        negative_prompt=negative_prompt,
        motion_score=args.motion_score,
        high_motion=args.high_motion,
        use_chi_prompt=use_chi_prompt,
    )
    
    # Save video
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prompt_slug = args.prompt[:50].replace(" ", "_").replace("/", "_")
    video_path = os.path.join(args.output_dir, f"sana_video_{timestamp}_{prompt_slug}.mp4")
    save_video(video, video_path, fps=16)
    
    # Extract frames if requested
    if args.extract_frames:
        frames_dir = os.path.join(args.output_dir, "frames")
        extract_frames(video, frames_dir, prefix=f"frame_{timestamp}")
    
    print("\n" + "=" * 80)
    print("✅ Inference completed successfully!")
    print(f"Video saved to: {video_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
