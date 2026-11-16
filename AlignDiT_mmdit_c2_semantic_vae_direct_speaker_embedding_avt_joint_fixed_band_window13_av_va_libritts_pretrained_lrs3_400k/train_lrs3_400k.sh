#!/usr/bin/env bash
# Validate this isolated LRS3 experiment; add --start for detached four-GPU training.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$project_dir/env.sh"
cd "$project_dir"
export PYTHONPATH="$project_dir/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
    "$project_dir/scripts/launch_lrs3_training.py" "$@"
