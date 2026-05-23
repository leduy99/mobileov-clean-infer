#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-"$REPO_ROOT/output/smoke_${STAMP}"}
GENERATED_DIR="${OUT_DIR}/generated"
UNDERSTAND_TXT="${OUT_DIR}/understanding.txt"
DEFAULT_GENERATION_CKPT="$REPO_ROOT/omni_ckpts/hf_mobile_ov/mobile_ov_135k_full.pt"

PROMPT=${PROMPT:-"a golden retriever running along a beach at sunset"}
QUESTION=${QUESTION:-"Describe the video in 2-3 sentences."}
STEPS=${STEPS:-24}
CFG_SCALE=${CFG_SCALE:-6.0}
NUM_FRAMES=${NUM_FRAMES:-9}
CHECKPOINT=${CHECKPOINT:-"$DEFAULT_GENERATION_CKPT"}

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Generation checkpoint not found: $CHECKPOINT" >&2
  echo "Copy the full Mobile-OV checkpoint into repo-local omni_ckpts/ or export CHECKPOINT=/abs/path/to/mobile_ov_full.pt" >&2
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
if [[ -n "${VIDEO_BACKBONE_CHECKPOINT_DIR:-}" ]]; then
  GEN_CMD+=(--checkpoint-dir "$VIDEO_BACKBONE_CHECKPOINT_DIR")
fi
if [[ -n "${SMOLVLM2_CKPT_PATH:-}" ]]; then
  GEN_CMD+=(--smolvlm2-ckpt-path "$SMOLVLM2_CKPT_PATH")
fi

"${GEN_CMD[@]}"

VIDEO_PATH=$(ls -1t "$GENERATED_DIR"/*.mp4 | head -n 1)
if [[ -z "${VIDEO_PATH}" ]]; then
  echo "No generated video found under ${GENERATED_DIR}" >&2
  exit 1
fi

python understand.py \
  --checkpoint "$CHECKPOINT" \
  --video "$VIDEO_PATH" \
  --prompt "$QUESTION" \
  --num-frames 8 \
  --max-new-tokens 96 | tee "$UNDERSTAND_TXT"

echo
echo "Smoke test finished."
echo "Generated video: $VIDEO_PATH"
echo "Understanding text: $UNDERSTAND_TXT"
