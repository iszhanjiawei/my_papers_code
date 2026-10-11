#!/usr/bin/env bash
# Prepare aligned GRID waveforms and the four frozen feature streams.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
exec "$python_bin" -u src/aligndit/script/misc/prepare_grid.py "$@"
