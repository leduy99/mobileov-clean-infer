# Architecture

This repository is a **clean self-contained inference repository**. It is not a
training repo, but it does carry the local generation code it needs so that
inference does not depend on a separate sibling checkout.

That means:

- the user-facing interface lives here
- the heavy generation backend is vendored locally
- the understanding path is intentionally kept simple and direct

The goal is to make inference easy to run without mixing it with the large
training codebase.

## High-level split

There are two separate paths in this clean repo:

1. **Generation**
2. **Understanding**

They intentionally use different backends.

## 1. Generation architecture

The generation path uses the local Mobile-OV generation stack inside this repo:

- wrapper here: `mobileov_infer/generate.py`
- local backend entrypoint: `tools/inference/test_q1_student_video.py`
- local SANA runtime: `tools/inference/sana_video_inference_fixed.py`
- local bridge/model code: `nets/...`
- local config files: `configs/sana_video_config/...`

### Generation dataflow

```text
Prompt text
  -> SmolVLM2 text path inside student bridge
  -> lexical-gated MCP projector
  -> SANA-video conditioning tokens
  -> SANA-video DiT denoising
  -> WAN VAE decode
  -> MP4 / PNG output
```

### More explicit view

```text
Prompt
  -> tokenizer
  -> SmolVLM2 hidden states
  -> semantic branch (last K layers)
  -> lexical branch (early hidden layer)
  -> scalar lexical gate
  -> fused prompt conditioning
  -> SANA-video diffusion model
  -> VAE decode
  -> generated video or image
```

### What is actually loaded

At inference time, the generation path loads:

- SANA-video backbone
- WAN VAE
- Mobile-OV bridge / projector weights
- optional DiT trainable state from the checkpoint, when present

### Default generation checkpoint in this clean repo

```text
omni_ckpts/hf_mobile_ov/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt
```

## 2. Understanding architecture

The understanding path does **not** reuse the large local research wrapper.
Instead, it uses Hugging Face SmolVLM2 directly.

- wrapper here: `mobileov_infer/understand.py`
- model id:
  `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`

### Understanding dataflow

```text
Image / video
  -> frame sampling (for video)
  -> Hugging Face AutoProcessor
  -> SmolVLM2-500M-Video-Instruct
  -> generated text response
```

### Why this path is separate

This was a deliberate design choice:

- generation in this project depends on the custom Mobile-OV + SANA backend
- understanding is much easier to keep stable by using the direct HF model

So the clean repo avoids pulling in older local VQA helper code and avoids
unnecessary dependency coupling.

## Repository architecture

The clean repo keeps the surface area small, but it now includes the local
runtime needed for generation:

```text
Mobile-OV-Infer-Clean/
  README.md
  ARCHITECTURE.md
  configs/
    sana_video_config/
  mobileov_infer/
    common.py
    generate.py
    understand.py
  nets/
    omni/
    smolvlm2/
    third_party/sana/
  tools/
    __init__.py
    inference/
      runtime_helpers.py
      sana_video_inference_fixed.py
      test_q1_student_video.py
  scripts/
    request_debug_tmux.sh
    generate.sh
    understand.sh
    smoke_test.sh
```

## Why the repo is still called "clean"

This repo is designed to stay clean at the workflow level:

- generation and understanding have short, stable entrypoints
- SLURM usage stays simple
- training utilities, dataset prep, and experiment notes stay out of the way

We did vendor the minimum local runtime needed for generation, because a thin
frontend that points to another repo is easy to break in practice and does not
meet the "self-contained" requirement.

So the design is:

- **clean inference workflow here**
- **local generation backend here**
- **source-of-truth understanding model from Hugging Face**

## Practical mental model

The easiest way to think about this repo is:

```text
This repo = simple CLI + SLURM-safe workflow
Generation model = local Mobile-OV + SANA inference stack
Understanding model = direct HF SmolVLM2
```
