#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-"$REPO_ROOT/output/smoke_${STAMP}"}
GENERATED_DIR="${OUT_DIR}/generated"
UNDERSTAND_TXT="${OUT_DIR}/understanding.txt"
DEFAULT_GENERATION_CKPT="$REPO_ROOT/omni_ckpts/hf_mobile_ov/stage1_joint_openvid_fullmobile_o_fulldit_diffonly_initlatest_bs64_v2_20260429_8gpu_60k.pt"
DEFAULT_SANA_CHECKPOINT_DIR="$REPO_ROOT/omni_ckpts/sana_video_2b_480p"
DEFAULT_SMOLVLM2_CKPT="$REPO_ROOT/omni_ckpts/smolvlm2_500m/smolvlm2_500m.pt"

PROMPT=${PROMPT:-"a golden retriever running along a beach at sunset"}
QUESTION=${QUESTION:-"Describe the video in 2-3 sentences."}
STEPS=${STEPS:-24}
CFG_SCALE=${CFG_SCALE:-6.0}
NUM_FRAMES=${NUM_FRAMES:-9}
CHECKPOINT=${CHECKPOINT:-"$DEFAULT_GENERATION_CKPT"}
SANA_CHECKPOINT_DIR=${SANA_CHECKPOINT_DIR:-"$DEFAULT_SANA_CHECKPOINT_DIR"}
SMOLVLM2_CKPT_PATH=${SMOLVLM2_CKPT_PATH:-"$DEFAULT_SMOLVLM2_CKPT"}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Generation checkpoint not found: $CHECKPOINT" >&2
  echo "Either copy weights into repo-local omni_ckpts/ or export CHECKPOINT=/abs/path/to/mobileov.pt" >&2
  exit 1
fi

if [[ ! -d "$SANA_CHECKPOINT_DIR" ]]; then
  echo "SANA checkpoint directory not found: $SANA_CHECKPOINT_DIR" >&2
  echo "Either copy weights into repo-local omni_ckpts/ or export SANA_CHECKPOINT_DIR=/abs/path/to/sana_video_2b_480p" >&2
  exit 1
fi

if [[ ! -f "$SMOLVLM2_CKPT_PATH" ]]; then
  echo "SmolVLM2 checkpoint not found: $SMOLVLM2_CKPT_PATH" >&2
  echo "Either copy weights into repo-local omni_ckpts/ or export SMOLVLM2_CKPT_PATH=/abs/path/to/smolvlm2_500m.pt" >&2
  exit 1
fi

mkdir -p "$GENERATED_DIR"

GEN_CMD=(
  python generate.py
  --prompt "$PROMPT"
  --output-dir "$GENERATED_DIR"
  --num-frames "$NUM_FRAMES"
  --steps "$STEPS"
  --cfg-scale "$CFG_SCALE"
  --seed 0
)

GEN_CMD+=(--checkpoint "$CHECKPOINT")
GEN_CMD+=(--checkpoint-dir "$SANA_CHECKPOINT_DIR")
GEN_CMD+=(--smolvlm2-ckpt-path "$SMOLVLM2_CKPT_PATH")

"${GEN_CMD[@]}"

VIDEO_PATH=$(ls -1t "$GENERATED_DIR"/*.mp4 | head -n 1)
if [[ -z "${VIDEO_PATH}" ]]; then
  echo "No generated video found under ${GENERATED_DIR}" >&2
  exit 1
fi

python understand.py \
  --video "$VIDEO_PATH" \
  --prompt "$QUESTION" \
  --num-frames 8 \
  --max-new-tokens 96 | tee "$UNDERSTAND_TXT"

echo
echo "Smoke test finished."
echo "Generated video: $VIDEO_PATH"
echo "Understanding text: $UNDERSTAND_TXT"
