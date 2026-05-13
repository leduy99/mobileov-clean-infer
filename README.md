# Mobile-OV Infer Clean

This repository is a clean inference frontend for:

- video and image generation
- image and video understanding

It is intentionally separated from the main training repo so we can run simple,
repeatable inference commands without mixing them with training scripts,
manifests, or experiment notes.

## Design

This repo keeps the user-facing interface clean, while reusing the tested model
backend from the sibling repository:

- backend repo: `../Omni-Video-smolvlm2`

That means we do **not** duplicate the full SANA code here. We only
wrap the tested generation entrypoint from the backend repo:

- generation backend:
  `tools/inference/test_q1_student_video.py`

For understanding, this clean repo now uses the Hugging Face
`HuggingFaceTB/SmolVLM2-500M-Video-Instruct` model directly, so that video
understanding stays simple and independent from backend research utilities.

## Important cluster rule

On the H200 machine, all GPU work must go through SLURM.

Do **not** run CUDA Python directly from a normal SSH shell.

## Quick start

From this repo:

```bash
bash scripts/request_debug_tmux.sh
```

That requests one debug GPU and opens a `tmux` session inside the allocation.

Inside that tmux session, you can run:

```bash
bash scripts/smoke_test.sh
```

The smoke test:

1. generates one short video
2. runs SmolVLM2 understanding on that video

Outputs go to:

```bash
output/smoke_YYYYMMDD_HHMMSS
```

## Generation

Simple example:

```bash
bash scripts/generate.sh \
  --prompt "a golden retriever running along a beach at sunset" \
  --num-frames 81 \
  --output-dir output/demo_generation
```

Useful overrides:

```bash
bash scripts/generate.sh \
  --checkpoint /abs/path/to/checkpoint.pt \
  --steps 24 \
  --cfg-scale 6.0 \
  --height 480 \
  --width 832 \
  --seed 0
```

Default generation checkpoint:

```bash
../Omni-Video-smolvlm2/omni_ckpts/hf_mobile_ov/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt
```

You can override it with:

```bash
export MOBILEOV_GENERATION_CKPT=/abs/path/to/another_checkpoint.pt
```

## Understanding

Text-only:

```bash
bash scripts/understand.sh \
  --prompt "Write a short poem about the moon."
```

Image understanding:

```bash
bash scripts/understand.sh \
  --image /abs/path/to/image.png \
  --prompt "Describe this image in detail."
```

Video understanding:

```bash
bash scripts/understand.sh \
  --video /abs/path/to/video.mp4 \
  --prompt "Describe the video in 2-3 sentences."
```

Default SmolVLM2 checkpoint:

```bash
HuggingFaceTB/SmolVLM2-500M-Video-Instruct
```

If your environment is missing the processor dependency used by SmolVLM2, install:

```bash
python -m pip install num2words
```

## Notes

- Run this repo from a shared filesystem path such as `/share_X/users/$USER`.
- The wrappers automatically set `PYTHONNOUSERSITE=1`.
- The wrappers automatically set `PYTHONPATH` to the backend repo before
  invoking the generation backend code.
- If your backend repo is not the sibling `../Omni-Video-smolvlm2`, set:

```bash
export MOBILEOV_BACKEND_REPO=/abs/path/to/Omni-Video-smolvlm2
```
