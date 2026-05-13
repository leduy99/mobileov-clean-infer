#!/usr/bin/env python3
"""
SANA-video inference script.
Downloads checkpoint (if needed) and generates video from text prompt.

Usage:
    python sana_video_inference.py \
        --prompt "a cat playing with a wool beside the fireside" \
        --output_dir output/sana_inference \
        --download_checkpoint
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

import argparse
import logging
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
import shutil

warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import yaml
from easydict import EasyDict
from tqdm import tqdm

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    print("Warning: OpenCV (cv2) not available. Video processing will be disabled.")
    CV2_AVAILABLE = False
    cv2 = None

# SANA imports
from nets.third_party.sana.diffusion.model.builder import (
    build_model,
    get_tokenizer_and_text_encoder,
    get_vae,
    vae_encode,
    vae_decode,
)
from nets.third_party.sana.diffusion.utils.config import SanaVideoConfig

# Try to import sampler - may not be available
try:
    from nets.third_party.sana.diffusion.utils.sampler import FlowMatchingSampler
    FLOW_MATCHING_SAMPLER_AVAILABLE = True
except ImportError:
    FLOW_MATCHING_SAMPLER_AVAILABLE = False
    print("Warning: FlowMatchingSampler not available. Will use simple sampling.")


def init_logging():
    """Initialize logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


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
        # Use huggingface-cli to download
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
        print("\nTrying alternative: manual download from HuggingFace website")
        print(f"URL: https://huggingface.co/{hf_model_id}")
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
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to EasyDict for easy access
    config = EasyDict(config_dict)
    return config


def load_sana_models(config, checkpoint_dir="omni_ckpts/sana_video_2b_480p", device="cuda:0", dtype=torch.bfloat16):
    """Load SANA-video model, VAE, and text encoder."""
    print("=" * 80)
    print("Loading SANA-video models...")
    print("=" * 80)
    
    # 1. Load text encoder
    print("\n[1/3] Loading text encoder...")
    text_encoder_name = config.text_encoder.text_encoder_name
    tokenizer, text_encoder = get_tokenizer_and_text_encoder(
        name=text_encoder_name,
        device=device
    )
    text_encoder.eval()
    print(f"✅ Text encoder loaded: {text_encoder_name}")
    
    # 2. Load VAE
    print("\n[2/3] Loading VAE...")
    vae_type = config.vae.vae_type
    vae_pretrained = config.vae.vae_pretrained
    
    # Handle HuggingFace path
    if vae_pretrained.startswith("hf://"):
        vae_path = vae_pretrained.replace("hf://", "")
        # Try to find in checkpoint dir or download
        vae_path = vae_path.replace("Efficient-Large-Model/SANA-Video_2B_480p/", f"{checkpoint_dir}/")
        # Remove hf:// prefix if still present
        if vae_path.startswith("hf://"):
            vae_path = vae_path.replace("hf://", "")
    else:
        vae_path = vae_pretrained
    
    # Make path absolute if relative
    if not os.path.isabs(vae_path):
        vae_path = os.path.abspath(vae_path)
    
    vae = get_vae(
        name=vae_type,
        model_path=vae_path,
        device=device,
        dtype=dtype,
        config=config.vae
    )
    # WanVAE is not a nn.Module, so no need to call eval()
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
        checkpoint_path = checkpoint_path.replace("Efficient-Large-Model/SANA-Video_2B_480p/", "omni_ckpts/sana_video_2b_480p/")
    else:
        checkpoint_path = load_from
    
    # Build model config
    from nets.third_party.sana.diffusion.utils.config import model_video_init_config
    model_cfg = model_video_init_config(config, latent_size=32)
    model_cfg['type'] = model_type
    
    # Build model
    diffusion_model = build_model(model_cfg)
    diffusion_model.eval()
    diffusion_model = diffusion_model.to(device).to(dtype)
    
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
        'diffusion_model': diffusion_model,
    }


@torch.no_grad()
def encode_text(tokenizer, text_encoder, prompt, device="cuda:0", max_length=300):
    """Encode text prompt to embeddings."""
    if tokenizer is None:
        # For Qwen2.5-VL
        assert hasattr(text_encoder, 'encode_text')
        y = text_encoder.encode_text([prompt], device=device)
        return y, None
    
    # Tokenize
    tokens = tokenizer(
        prompt,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).to(device)
    
    # Encode
    text_embeddings = text_encoder(
        tokens.input_ids,
        attention_mask=tokens.attention_mask
    ).last_hidden_state
    
    # SANA expects shape: (N, 1, seq_len, hidden_dim)
    text_embeddings = text_embeddings.unsqueeze(1)
    
    return text_embeddings, tokens


def simple_flow_matching_sampling(
    diffusion_model,
    text_embeddings,
    shape,
    num_steps=50,
    device="cuda:0",
    dtype=torch.bfloat16,
    cfg_scale=7.0,
    flow_shift=7.0,
):
    """
    Simple flow matching sampling (approximation).
    
    Args:
        shape: (B, C, T, H, W) latent shape
    """
    print(f"Running flow matching sampling with {num_steps} steps...")
    
    B, C, T, H, W = shape
    
    # Initialize noise
    x = torch.randn(shape, device=device, dtype=dtype)
    
    # Timesteps (logit-normal schedule)
    # For flow matching, we use a simple linear schedule
    timesteps = torch.linspace(0, 1, num_steps + 1, device=device)
    
    for i in tqdm(range(num_steps), desc="Sampling"):
        t = timesteps[i].unsqueeze(0).repeat(B)
        
        # Predict velocity
        if cfg_scale > 1.0:
            # Classifier-free guidance
            # Unconditioned
            uncond_embeddings = torch.zeros_like(text_embeddings)
            velocity_uncond = diffusion_model(
                x, t, uncond_embeddings, mask=None
            )
            
            # Conditioned
            velocity_cond = diffusion_model(
                x, t, text_embeddings, mask=None
            )
            
            # CFG combination
            velocity = velocity_uncond + cfg_scale * (velocity_cond - velocity_uncond)
        else:
            velocity = diffusion_model(x, t, text_embeddings, mask=None)
        
        # Euler step
        dt = timesteps[i+1] - timesteps[i]
        x = x + velocity * dt
    
    return x


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
):
    """Generate video from text prompt."""
    print("=" * 80)
    print(f"Generating video from prompt: {prompt}")
    print(f"Resolution: {width}x{height}, Frames: {num_frames}")
    print(f"Inference steps: {num_inference_steps}, CFG scale: {cfg_scale}")
    print(
        "Sampling parameters: backend=legacy seed=%d num_steps=%d cfg_scale=%.4f flow_shift=%.4f "
        "num_frames=%d height=%d width=%d device=%s"
        % (
            int(seed),
            int(num_inference_steps),
            float(cfg_scale),
            7.0,
            int(num_frames),
            int(height),
            int(width),
            str(device),
        )
    )
    print("=" * 80)
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Encode text
    print("\n[1/4] Encoding text prompt...")
    text_embeddings, tokens = encode_text(
        models['tokenizer'],
        models['text_encoder'],
        prompt,
        device=device
    )
    print(f"✅ Text embeddings shape: {text_embeddings.shape}")
    
    # Prepare latent shape
    vae_downsample = 8  # Default for WanVAE
    latent_h = height // vae_downsample
    latent_w = width // vae_downsample
    latent_c = 16  # From config
    latent_shape = (1, latent_c, num_frames, latent_h, latent_w)
    print(f"\n[2/4] Latent shape: {latent_shape}")
    
    # Sample from diffusion model
    print("\n[3/4] Running diffusion sampling...")
    
    # Simple sampling (flow matching approximation)
    latents = simple_flow_matching_sampling(
        models['diffusion_model'],
        text_embeddings,
        latent_shape,
        num_steps=num_inference_steps,
        device=device,
        dtype=dtype,
        cfg_scale=cfg_scale,
        flow_shift=7.0,
    )
    
    # Decode with VAE
    print("\n[4/4] Decoding latents to video...")
    vae_type = "WanVAE"  # From config
    frames_list = []
    
    # Decode frame by frame (or in batches)
    batch_size = 8  # Decode 8 frames at a time
    for t_start in range(0, num_frames, batch_size):
        t_end = min(t_start + batch_size, num_frames)
        latent_batch = latents[:, :, t_start:t_end, :, :]
        
        # Reshape to (B, C, H, W) for VAE decode
        B, C, T, H, W = latent_batch.shape
        latent_batch = latent_batch.permute(0, 2, 1, 3, 4).contiguous()
        latent_batch = latent_batch.view(B * T, C, H, W)
        
        # Decode
        frames_batch = vae_decode(vae_type, models['vae'], latent_batch)
        
        # Reshape back
        frames_batch = frames_batch.view(B, T, *frames_batch.shape[1:])
        frames_list.append(frames_batch.cpu())
    
    video = torch.cat(frames_list, dim=1)  # (B, T, C, H, W)
    video = video.squeeze(0)  # (T, C, H, W)
    
    # Normalize to [0, 255]
    video = (video + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    video = torch.clamp(video, 0, 1)
    video = (video * 255).byte()
    
    # Convert to numpy
    video = video.permute(0, 2, 3, 1).numpy()  # (T, H, W, C)
    
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
    parser = argparse.ArgumentParser(description="SANA-video inference")
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for video generation")
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
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--extract_frames", action="store_true",
                       help="Extract frames from generated video")
    
    args = parser.parse_args()
    
    init_logging()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download checkpoint if requested
    if args.download_checkpoint:
        print("=" * 80)
        print("Downloading checkpoint...")
        print("=" * 80)
        download_checkpoint(local_dir=args.checkpoint_dir)
    
    # Load config
    config = load_config_file(args.config)
    
    # Update checkpoint paths in config if needed
    if config.model.load_from.startswith("hf://"):
        config.model.load_from = config.model.load_from.replace(
            "Efficient-Large-Model/SANA-Video_2B_480p/",
            f"{args.checkpoint_dir}/"
        )
    if config.vae.vae_pretrained.startswith("hf://"):
        config.vae.vae_pretrained = config.vae.vae_pretrained.replace(
            "Efficient-Large-Model/SANA-Video_2B_480p/",
            f"{args.checkpoint_dir}/"
        )
    
    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    models = load_sana_models(config, checkpoint_dir=args.checkpoint_dir, device=str(device), dtype=dtype)
    
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
        dtype=dtype,
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
