#!/usr/bin/env bash

# Source this file from an interactive shell:
#   source scripts/activate_mobileov.sh
#
# On this cluster, tmux panes can keep the base conda python ahead of the
# target env in PATH even after `conda activate`. Re-prepending CONDA_PREFIX/bin
# keeps `python` pointed at the real mobileov env.

source /share_0/conda/etc/profile.d/conda.sh
conda activate mobileov
export PATH="$CONDA_PREFIX/bin:$PATH"
export PYTHONNOUSERSITE=1
