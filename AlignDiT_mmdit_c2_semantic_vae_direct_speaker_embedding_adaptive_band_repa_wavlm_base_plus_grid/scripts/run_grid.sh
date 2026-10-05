#!/usr/bin/env bash
# Complete, auditable preprocessing followed by the independent GRID run.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
cache_root="${GRID_CACHE_ROOT:-${ROOT_PREFIX}/zjw524/projects/data/GRID_mmdit_svae}"
mkdir -p logs
exec 8>logs/grid_pipeline.lock
flock -n 8 || { echo 'A GRID pipeline is already active in this copy' >&2; exit 1; }
if [[ "${WAIT_FOR_EXISTING_PREP:-0}" != 1 ]]; then
    bash scripts/prepare_grid.sh --output "$cache_root" > logs/prepare_grid_full.log 2>&1
fi
[[ -f "$cache_root/.prepare.lock" ]] || { echo "No preparation has started at $cache_root" >&2; exit 1; }
printf 'Waiting for complete GRID preprocessing and audit at %s\n' "$cache_root"
# The preparer holds this lock through the final checksum audit. Failure releases
# the lock too, so require the bound completion contract before starting training.
flock -x "$cache_root/.prepare.lock" "$python_bin" - "$cache_root" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
contract_path = root / "data_contract.json"
contract = json.loads(contract_path.read_text())
complete = json.loads((root / "complete.json").read_text())
if not (
    contract.get("dataset") == "GRID"
    and contract.get("complete") is True
    and contract.get("selection", {}).get("mode") == "full"
    and contract.get("split_counts") == {"train": 29557, "val": 3281}
    and complete.get("complete") is True
    and complete.get("source_and_feature_sha256_verified") is True
    and complete.get("contract_sha256") == hashlib.sha256(contract_path.read_bytes()).hexdigest()
):
    raise RuntimeError("Full GRID cache did not pass its completion audit")
print("GRID preprocessing and checksum audit passed; launching training", flush=True)
PY
exec bash scripts/train_grid.sh "datasets.cache_root=$cache_root" "$@" > logs/train_grid.log 2>&1
