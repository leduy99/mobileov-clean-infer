# Mobile-OV Clean Infer

This repository is a **self-contained inference repo** for two things:

- Mobile-OV generation
- SmolVLM2 understanding

It is intentionally separate from the training repo. The goal here is not to
carry every research branch. The goal is to keep one small codebase that an
engineer can:

- run under SLURM
- read end to end
- trace from CLI entrypoint to model code
- use as a starting point for porting to another runtime such as mobile or
  TensorFlow

The clean repo therefore makes a few strong choices:

- one top-level full model class: `nets/mobile_ov/mobile_ov_model.py`
- one active generation path: **SmolVLM2 -> lexical-gated Mobile-OV bridge -> video diffusion backbone**
- one active understanding path: **local SmolVLM2 PyTorch model**
- no sibling-repo dependency
- no `python -m mobileov_infer...` wrapper package
- no alternative bridge backbones such as Gemma/Qwen in the active path
- no generation-time LoRA branches or multi-backbone inference branches

Architecture and file map:

- [ARCHITECTURE.md](ARCHITECTURE.md)

Primary Python API:

- `nets/mobile_ov/mobile_ov_model.py`
- class: `MobileOVModel`

## Environment

```bash
source scripts/activate_mobileov.sh
```

If your environment is missing the SmolVLM2 processor dependency:

```bash
python -m pip install num2words
```

## Important cluster rule

On the H200 cluster, all GPU work must go through SLURM.

Do not run CUDA Python directly from a normal SSH shell.

## Use The Full Checkpoint

For normal sharing and conversion, give colleagues this repo plus one checkpoint
file:

```text
mobile_ov_135k_full.pt
```

This file is the complete Mobile-OV inference checkpoint. It includes:

- SmolVLM2 model weights
- SmolVLM2 tokenizer and processor assets
- Mobile-OV bridge and lexical gate weights
- merged video DiT generator weights
- VAE weights

It does **not** require separate SANA-video or SmolVLM2 checkpoint files at
inference time.

Recommended local layout:

```bash
mkdir -p omni_ckpts/hf_mobile_ov
cp /path/to/mobile_ov_135k_full.pt omni_ckpts/hf_mobile_ov/mobile_ov_135k_full.pt
```

If the checkpoint is hosted on Hugging Face, the equivalent download command is:

```bash
mkdir -p omni_ckpts/hf_mobile_ov
huggingface-cli download leduy99/Mobile-OV mobile_ov_135k_full.pt \
  --local-dir omni_ckpts/hf_mobile_ov
```

If the checkpoint is stored elsewhere, pass it explicitly:

```bash
CHECKPOINT=/abs/path/to/mobile_ov_135k_full.pt bash scripts/smoke_test.sh
```

The first run extracts the embedded sub-checkpoints into:

```text
~/.cache/mobile_ov/bundles/
```

To use a different cache location:

```bash
export MOBILEOV_BUNDLE_CACHE=/share_4/users/$USER/mobile_ov_cache
```

## Quick start

Request one debug GPU and open `tmux` inside the allocation:

```bash
bash scripts/request_debug_tmux.sh
```

Inside that tmux session:

```bash
source scripts/activate_mobileov.sh
bash scripts/smoke_test.sh
```

Why use the helper script?

- it activates the `mobileov` env by name
- it re-prepends `"$CONDA_PREFIX/bin"` to `PATH`
- that avoids a tmux-specific shell quirk where `python` can still point at the
  base conda install even after `conda activate`

The smoke test:

1. generates one short video with Mobile-OV
2. runs local SmolVLM2 understanding on that video

Current smoke-test defaults:

- generation steps: `24`
- cfg scale: `6.0`
- generation frames: `9`

The smoke test is intentionally lighter than a normal benchmark run. It is
meant to verify that:

- the generation path still works end to end
- the understanding path still works end to end
- SLURM + tmux workflow is healthy

By default it looks for a repo-local full Mobile-OV checkpoint at
`omni_ckpts/hf_mobile_ov/mobile_ov_135k_full.pt`. If the checkpoint lives
elsewhere, override it explicitly:

```bash
CHECKPOINT=/abs/path/to/mobile_ov_135k_full.pt bash scripts/smoke_test.sh
```

Outputs are written under:

```text
output/smoke_YYYYMMDD_HHMMSS/
```

## Generation

This repo supports one generation architecture only:

- Mobile-OV lexical-gated bridge
- SmolVLM2-500M text path
- video diffusion backbone base model
- merged full-DiT generator weights from the full Mobile-OV checkpoint

That means the generation backend is intentionally **not** a general experiment
launcher. It is a small, readable implementation of the exact path used by the
current joint full-DiT checkpoint family. Research versioning still lives in checkpoint
names. The clean repo itself uses the simpler architecture name `Mobile-OV`.

User-facing entrypoint:

```bash
python generate.py \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --prompt "a golden retriever running along a beach at sunset" \
  --num-frames 81 \
  --output-dir output/demo_generation
```

Shell convenience wrapper:

```bash
bash scripts/generate.sh \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --prompt "a golden retriever running along a beach at sunset" \
  --num-frames 81 \
  --output-dir output/demo_generation
```

Default generation settings:

- steps: `24`
- cfg scale: `6.0`
- seed: `0`
- dtype: `bf16`

Useful overrides:

```bash
python generate.py \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --steps 24 \
  --cfg-scale 6.0 \
  --height 480 \
  --width 832 \
  --seed 0
```

Direct Python API:

```python
from nets.mobile_ov import MobileOVModel

model = MobileOVModel(
    generation_ckpt_path="/abs/path/to/mobile_ov_135k_full.pt",
    device="cuda:0",
    dtype="bf16",
)
video_path = model.generate_video(
    prompt="a golden retriever running along a beach at sunset",
    output_dir="output/demo_generation",
    num_frames=81,
    steps=24,
    cfg_scale=6.0,
)
```

## Understanding

User-facing entrypoint:

```bash
python understand.py \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --video /abs/path/to/video.mp4 \
  --prompt "Describe the video in 2-3 sentences."
```

Text-only:

```bash
python understand.py \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --prompt "Write a short poem about the moon."
```

Image understanding:

```bash
python understand.py \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --image /abs/path/to/image.png \
  --prompt "Describe this image in detail."
```

Important note:

- the **understanding model itself** is local PyTorch code in `nets/smolvlm2/`
- `transformers` is still used for tokenizer/processor convenience
- the repo does **not** use `AutoModel...` as the model runtime
- this means engineers can inspect the model implementation locally instead of
  treating the model as a black box hidden inside a library

Default full Mobile-OV checkpoint:

```text
omni_ckpts/hf_mobile_ov/mobile_ov_135k_full.pt
```

Direct Python API:

```python
from nets.mobile_ov import MobileOVModel

model = MobileOVModel(
    generation_ckpt_path="/abs/path/to/mobile_ov_135k_full.pt",
    device="cuda:0",
    dtype="bf16",
)
text = model.understand_video(
    video_path="/abs/path/to/video.mp4",
    prompt="Describe the video in 2-3 sentences.",
    num_frames=8,
)
print(text)
```

## Expected checkpoint layout

For model conversion, use a complete Mobile-OV checkpoint. This packs SmolVLM2,
the Mobile-OV bridge, the merged video DiT weights, and the VAE into one `.pt`
file:

```bash
python tools/package_mobile_ov_full_checkpoint.py \
  --mobile-ov-checkpoint /abs/path/to/mobile_ov_135k.pt \
  --smolvlm2-checkpoint /abs/path/to/smolvlm2_500m.pt \
  --video-backbone-checkpoint /abs/path/to/SANA_Video_2B_480p.pth \
  --vae-checkpoint /abs/path/to/Wan2.1_VAE.pth \
  --tokenizer-assets-dir /abs/path/to/SmolVLM2-500M-Video-Instruct \
  --output /abs/path/to/mobile_ov_135k_full.pt
```

Generation can then run from only that checkpoint:

```bash
python generate.py \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --prompt "a golden retriever running along a beach at sunset" \
  --output-dir output/demo_generation
```

Understanding can use the same full checkpoint to extract the embedded SmolVLM2
weights:

```bash
python understand.py \
  --checkpoint /abs/path/to/mobile_ov_135k_full.pt \
  --prompt "Describe this video in one sentence." \
  --video /abs/path/to/video.mp4
```

For smaller research artifacts, Mobile-OV and SmolVLM2 can also be packaged
without embedding the public video backbone:

```bash
python tools/package_mobile_ov_bundle.py \
  --mobile-ov-checkpoint /abs/path/to/mobile_ov_135k.pt \
  --smolvlm2-checkpoint /abs/path/to/smolvlm2_500m.pt \
  --output /abs/path/to/mobile_ov_135k_smolvlm2_bundle.pt
```

Then generation only needs the bundle plus the public SANA-video backbone:

```bash
python generate.py \
  --checkpoint /abs/path/to/mobile_ov_135k_smolvlm2_bundle.pt \
  --checkpoint-dir /abs/path/to/sana_video_2b_480p \
  --prompt "a golden retriever running along a beach at sunset" \
  --output-dir output/demo_generation
```

That smaller bundle still needs `--checkpoint-dir` because it intentionally does
not include the video backbone weights.

The legacy unbundled layout is:

```text
omni_ckpts/
  hf_mobile_ov/
    stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt
  sana_video_2b_480p/
    checkpoints/SANA_Video_2B_480p.pth
    vae/Wan2.1_VAE.pth
  smolvlm2_500m/
    smolvlm2_500m.pt
```

If your checkpoints live elsewhere, pass them explicitly:

```bash
python generate.py --checkpoint /abs/path/to/mobile_ov_135k_full.pt ...
python understand.py --checkpoint /abs/path/to/mobile_ov_135k_full.pt ...
```

## Active code surface

If you only want to understand the repo, start here:

- `generate.py`
- `understand.py`
- `nets/mobile_ov/mobile_ov_model.py`
- `nets/mobile_ov/mobile_ov_bridge.py`
- `tools/inference/video_backbone_runtime.py`
- `nets/mobile_ov/adapter.py`
- `nets/mobile_ov/smolvlm2_vision_head.py`
- `nets/smolvlm2/`

Everything else exists only because the active path still needs it at runtime,
most notably the vendored video diffusion stack under `nets/third_party/video_backbone/`.
