#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec env CUDA_VISIBLE_DEVICES="${PREP_GPU:-0}" "$python_bin" -u \
    src/aligndit/script/misc/prepare_grid_mmdit.py "$@"
