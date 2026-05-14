#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

PARTITION=${PARTITION:-debug}
GPUS=${GPUS:-1}
TIME_LIMIT=${TIME_LIMIT:-02:00:00}
CPUS_PER_TASK=${CPUS_PER_TASK:-16}
MEMORY=${MEMORY:-128G}
SESSION_NAME=${SESSION_NAME:-mobileov_clean}

exec srun \
  --partition="${PARTITION}" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --pty bash -lc "cd '${REPO_ROOT}' && tmux new -A -s '${SESSION_NAME}'"
