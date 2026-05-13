# Architecture

This repository is a **clean inference frontend**, not a full training or model
development repository.

That means:

- the user-facing interface lives here
- the heavy generation backend is reused from the sibling repository
- the understanding path is intentionally kept simple and direct

The goal is to make inference easy to run without mixing it with the large
training codebase.

## High-level split

There are two separate paths in this clean repo:

1. **Generation**
2. **Understanding**

They intentionally use different backends.

## 1. Generation architecture

The generation path uses the tested Mobile-OV checkpoint through the sibling
backend repository:

- backend repo: `../Omni-Video-smolvlm2`
- wrapper here: `mobileov_infer/generate.py`
- backend entrypoint:
  `tools/inference/test_q1_student_video.py`

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

At inference time, the backend path loads:

- SANA-video backbone
- WAN VAE
- Mobile-OV bridge / projector weights
- optional DiT trainable state from the checkpoint, when present

### Default generation checkpoint in this clean repo

```text
../Omni-Video-smolvlm2/omni_ckpts/hf_mobile_ov/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt
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

The clean repo itself is intentionally small:

```text
Mobile-OV-Infer-Clean/
  README.md
  ARCHITECTURE.md
  mobileov_infer/
    common.py
    generate.py
    understand.py
  scripts/
    request_debug_tmux.sh
    generate.sh
    understand.sh
    smoke_test.sh
```

## Why the full model code is not duplicated here

This repo is designed to stay clean and low-risk.

If we copied the full model implementation here, we would immediately create:

- code drift
- duplicate bug-fix work
- confusion about which repo is the source of truth

So the design is:

- **clean frontend here**
- **source-of-truth generation backend in `Omni-Video-smolvlm2`**
- **source-of-truth understanding model from Hugging Face**

## Practical mental model

The easiest way to think about this repo is:

```text
This repo = simple CLI + SLURM-safe workflow
Generation model = sibling backend repo
Understanding model = direct HF SmolVLM2
```

If you want the full research / training / internal model implementation, look
at:

```text
../Omni-Video-smolvlm2
```
