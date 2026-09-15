#!/usr/bin/env bash
# Complete the resumable weight -> full feature cache -> readiness pipeline.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
export PYTHONPATH="$project_root/src"
export PYTHONUNBUFFERED=1
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
mkdir -p logs
bash scripts/download_synchformer.sh
bash scripts/extract_synchformer_multigpu.sh "$@"
"$python_bin" -u scripts/preflight_synchformer.py --report logs/synchformer_preflight.json
