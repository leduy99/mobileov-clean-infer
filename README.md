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

- one active generation path: **SmolVLM2 -> lexical-gated Mobile-OV bridge -> SANA-video**
- one active understanding path: **local SmolVLM2 PyTorch model**
- no sibling-repo dependency
- no `python -m mobileov_infer...` wrapper package
- no alternative bridge backbones such as Gemma/Qwen in the active path

Architecture and file map:

- [ARCHITECTURE.md](ARCHITECTURE.md)

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

By default it looks for repo-local weights under `omni_ckpts/`. If your weights
live elsewhere, override them explicitly:

```bash
CHECKPOINT=/abs/path/to/mobileov.pt \
SANA_CHECKPOINT_DIR=/abs/path/to/sana_video_2b_480p \
SMOLVLM2_CKPT_PATH=/abs/path/to/smolvlm2_500m.pt \
bash scripts/smoke_test.sh
```

Outputs are written under:

```text
output/smoke_YYYYMMDD_HHMMSS/
```

## Generation

User-facing entrypoint:

```bash
python generate.py \
  --prompt "a golden retriever running along a beach at sunset" \
  --num-frames 81 \
  --output-dir output/demo_generation
```

Shell convenience wrapper:

```bash
bash scripts/generate.sh \
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
  --checkpoint /abs/path/to/checkpoint.pt \
  --steps 24 \
  --cfg-scale 6.0 \
  --height 480 \
  --width 832 \
  --seed 0
```

Default generation checkpoint:

```text
omni_ckpts/hf_mobile_ov/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt
```

## Understanding

User-facing entrypoint:

```bash
python understand.py \
  --video /abs/path/to/video.mp4 \
  --prompt "Describe the video in 2-3 sentences."
```

Text-only:

```bash
python understand.py \
  --prompt "Write a short poem about the moon."
```

Image understanding:

```bash
python understand.py \
  --image /abs/path/to/image.png \
  --prompt "Describe this image in detail."
```

Important note:

- the **understanding model itself** is local PyTorch code in `nets/smolvlm2/`
- `transformers` is still used for tokenizer/processor convenience
- the repo does **not** use `AutoModel...` as the model runtime
- this means engineers can inspect the model implementation locally instead of
  treating the model as a black box hidden inside a library

Default local SmolVLM2 checkpoint:

```text
omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt
```

## Expected checkpoint layout

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
python generate.py --checkpoint /abs/path/to/mobileov.pt ...
python understand.py --ckpt-path /abs/path/to/smolvlm2_500m.pt ...
```

## Active code surface

If you only want to understand the repo, start here:

- `generate.py`
- `understand.py`
- `tools/inference/test_q1_student_video.py`
- `tools/inference/sana_video_inference_fixed.py`
- `nets/omni/modules/sana_prompt_bridge.py`
- `nets/omni/modules/adapter.py`
- `nets/omni/modules/smolvlm2_vision_head.py`
- `nets/smolvlm2/`

Everything else exists only because the active path still needs it at runtime,
most notably the vendored SANA diffusion stack under `nets/third_party/sana/`.
