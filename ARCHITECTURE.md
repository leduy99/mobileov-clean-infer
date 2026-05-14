# Architecture

This document maps the **actual active Mobile-OV inference path** in this clean
repo.

The emphasis is practical:

- what the model is made of
- which files implement each block
- which file is the real entrypoint
- which parts are custom Mobile-OV code
- which parts are vendored diffusion-backbone runtime code

## 1. High-level structure

There are two independent paths in this repo:

1. **Generation**
2. **Understanding**

Generation uses the full Mobile-OV + video diffusion backbone stack.

Understanding uses the local SmolVLM2 model only.

That split is deliberate. It keeps generation faithful to the trained Mobile-OV
checkpoint while keeping understanding simple and readable.

Important simplification:

- generation is **single-path**
- the clean repo does not try to support legacy bridge types, Gemma/Qwen
  backbones, LoRA inference branches, or dual-text inference
- `tools/inference/mobile_ov_generate.py` is intentionally written around the
  current Mobile-OV lexical-gated checkpoint family
- research versioning stays in checkpoint names; the clean repo uses the simpler
  architecture name `Mobile-OV`

## 2. Mobile-OV generation: block diagram

```text
Prompt text
  -> SmolVLM2 text model
  -> Mobile-OV lexical-gated bridge
  -> backbone conditioning tokens
  -> video diffusion denoising
  -> WAN VAE decode
  -> MP4 / PNG output
```

More explicitly:

```text
Prompt
  -> tokenizer / text preprocessing
  -> SmolVLM2 hidden states
  -> semantic branch (late hidden layers)
  -> lexical branch (early hidden layer)
  -> learned scalar lexical gate
  -> fused prompt tokens
  -> video diffusion model
  -> WAN VAE
  -> decoded video frames
```

## 3. Generation code map

### 3.1 User-facing entrypoint

- `generate.py`

This is the clean CLI entrypoint. It only exposes the generation settings we
actually use for inference, such as:

- Mobile-OV checkpoint path
- backbone checkpoint directory
- SmolVLM2 checkpoint path
- prompt
- frame count
- steps
- CFG

It does **not** spawn another repo or shell out to a wrapper package.

### 3.2 Main generation implementation

- `tools/inference/mobile_ov_generate.py`

This is the main Mobile-OV generation implementation in this repo.

Responsibilities:

- load the Mobile-OV checkpoint
- validate that the checkpoint matches the supported architecture
- build the SmolVLM2-based lexical-gated bridge
- prepare prompt embeddings
- load the DiT trainable state from checkpoint
- call the vendored backbone runtime for sampling
- decode and save output

This file is where the trained Mobile-OV checkpoint is connected to the SANA
runtime, and it is intentionally the single most important file to read for the
generation path.

### 3.3 Bridge implementation

- `nets/mobile_ov/mobile_ov_bridge.py`

This file contains the core Mobile-OV bridge logic.

Key ideas implemented here:

- load local SmolVLM2
- collect hidden states from the text model
- fuse late semantic features
- inject early lexical features
- apply the learned scalar lexical gate
- project into the backbone conditioning space

This is the main custom Mobile-OV code that differentiates the model from
plain backbone inference.

### 3.4 Supporting bridge modules

- `nets/mobile_ov/adapter.py`
- `nets/mobile_ov/smolvlm2_vision_head.py`

These files implement helper modules used by the bridge:

- adapter blocks
- optional resampling / query compression path

Even if a given checkpoint does not rely heavily on the vision-head branch,
these files are part of the active code path and are kept here for completeness.

### 3.5 Diffusion backbone runtime

- `tools/inference/video_backbone_runtime.py`

This is the clean diffusion-backbone runtime used by the repo.

Responsibilities:

- load backbone config
- load the base video diffusion model
- load WAN VAE
- prepare aspect ratio and latent shapes
- run flow-matching / DPM sampling
- decode latents into frames
- save MP4

This file is **runtime glue** around the vendored backbone implementation.

### 3.6 Vendored backbone code

- `nets/third_party/video_backbone/`

This directory contains the vendored backbone implementation that the runtime uses.

Important subareas:

- `nets/third_party/video_backbone/diffusion/model/`
- `nets/third_party/video_backbone/diffusion/scheduler/`
- `nets/third_party/video_backbone/diffusion/longsana/`

These files are not Mobile-OV-specific, but they are needed because Mobile-OV
generation depends on the vendored diffusion backbone.

## 4. Understanding code map

### 4.1 User-facing entrypoint

- `understand.py`

This is the clean CLI entrypoint for understanding.

Responsibilities:

- load a local SmolVLM2 checkpoint
- sample frames from video inputs
- build tokenizer/processor inputs
- run text generation on the local SmolVLM2 model

### 4.2 Local SmolVLM2 model code

- `nets/smolvlm2/load_smolvlm2.py`
- `nets/smolvlm2/modeling_smolvlm2.py`
- `nets/smolvlm2/architecture_smolvlm2.py`
- `nets/smolvlm2/config_smolvlm2.py`

This is the local SmolVLM2 implementation used by `understand.py`.

File roles:

- `load_smolvlm2.py`
  Loads the converted `.pt` checkpoint into the local model class.
- `modeling_smolvlm2.py`
  Public wrapper class for loading and generation.
- `architecture_smolvlm2.py`
  Core network architecture.
- `config_smolvlm2.py`
  Minimal config objects needed to reconstruct the model.

Important design choice:

- the **model implementation is local**
- `transformers` is used only for tokenizer/processor convenience
- the repo does **not** rely on `AutoModel...` as the SmolVLM2 runtime

That keeps the model readable for engineers who need to port it to another
runtime.

## 5. Checkpoint contract

The Mobile-OV generation path expects three kinds of weights:

### 5.1 Mobile-OV checkpoint

Typical file:

```text
omni_ckpts/hf_mobile_ov/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt
```

This checkpoint provides:

- `student_state`
- `infer_hints`
- optional `dit_trainable_state`

`tools/inference/mobile_ov_generate.py` reads these fields and reconstructs
the bridge plus any trainable DiT deltas.

### 5.2 Base backbone checkpoint directory

Typical directory:

```text
omni_ckpts/sana_video_2b_480p/
```

This directory provides:

- base video diffusion weights
- WAN VAE weights

### 5.3 Local SmolVLM2 checkpoint

Typical file:

```text
omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt
```

This file is used in two places:

- as the text model inside the Mobile-OV bridge
- as the standalone local model for understanding

## 6. What was intentionally removed

This clean repo intentionally removes or disables code that is not part of the
active path:

- wrapper package entrypoints
- sibling-repo dispatch
- alternative bridge backbones such as Qwen/Gemma
- old Omni training/integrated model classes
- old dataset loader code
- the LLaVA tree
- the legacy backbone inference script

That is why the repo is smaller and easier to trace than the training repo.

## 7. Minimal reading order

If someone new needs to understand the repo quickly, read in this order:

1. `README.md`
2. `generate.py`
3. `tools/inference/mobile_ov_generate.py`
4. `nets/mobile_ov/mobile_ov_bridge.py`
5. `tools/inference/video_backbone_runtime.py`
6. `understand.py`
7. `nets/smolvlm2/architecture_smolvlm2.py`

That order gives the shortest path from CLI to model internals.
