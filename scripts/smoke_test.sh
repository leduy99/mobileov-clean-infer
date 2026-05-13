#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$REPO_ROOT"

STAMP=$(date +%Y%m%d_%H%M%S)
OUT_DIR=${OUT_DIR:-"$REPO_ROOT/output/smoke_${STAMP}"}
GENERATED_DIR="${OUT_DIR}/generated"
UNDERSTAND_TXT="${OUT_DIR}/understanding.txt"

PROMPT=${PROMPT:-"a golden retriever running along a beach at sunset"}
QUESTION=${QUESTION:-"Describe the video in 2-3 sentences."}

mkdir -p "$GENERATED_DIR"

python -m mobileov_infer.generate \
  --prompt "$PROMPT" \
  --output-dir "$GENERATED_DIR" \
  --num-frames 17 \
  --steps 8 \
  --cfg-scale 6.0 \
  --seed 0

VIDEO_PATH=$(ls -1t "$GENERATED_DIR"/*.mp4 | head -n 1)
if [[ -z "${VIDEO_PATH}" ]]; then
  echo "No generated video found under ${GENERATED_DIR}" >&2
  exit 1
fi

python -m mobileov_infer.understand \
  --video "$VIDEO_PATH" \
  --prompt "$QUESTION" \
  --num-frames 8 \
  --max-new-tokens 96 | tee "$UNDERSTAND_TXT"

echo
echo "Smoke test finished."
echo "Generated video: $VIDEO_PATH"
echo "Understanding text: $UNDERSTAND_TXT"
