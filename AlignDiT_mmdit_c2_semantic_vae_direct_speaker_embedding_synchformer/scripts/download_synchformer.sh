#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
model_dir="${SYNCHFORMER_MODEL_DIR:-${ROOT_PREFIX}/zjw524/projects/data/pretrained_models/HunyuanVideo-Foley}"
checkpoint="$model_dir/synchformer_state_dict.pth"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "$model_dir"
if [[ ! -f "$checkpoint" ]]; then
  if [[ "${DOWNLOAD_WORKERS:-16}" -gt 1 ]]; then
    "$python_bin" -u "$project_root/scripts/download_synchformer_ranges.py" "$checkpoint" \
        --workers "${DOWNLOAD_WORKERS:-16}"
  else
    "$python_bin" -m huggingface_hub.cli.hf download tencent/HunyuanVideo-Foley \
        synchformer_state_dict.pth \
        --revision 3abd4e833b95b8db0fc9c687afc52483a48e9a97 \
        --local-dir "$model_dir"
  fi
fi
"$python_bin" - "$checkpoint" <<'PY'
import hashlib
import sys
from pathlib import Path
p = Path(sys.argv[1])
h = hashlib.sha256()
with p.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
expected = '8aff082f2df5c3bc52759db0c865c7ee772ae6400b860d1b7e90413f2defb67c'
if h.hexdigest() != expected:
    raise SystemExit(f'Checkpoint SHA256 mismatch: {p}; expected {expected}, got {h.hexdigest()}')
print(f'Verified Synchformer: {p} ({p.stat().st_size} bytes), SHA256={h.hexdigest()}', flush=True)
PY
