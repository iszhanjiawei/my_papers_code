#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source env.sh
export PYTHONPATH="$project_root/src" PYTHONUNBUFFERED=1
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
data="${ROOT_PREFIX}/zjw524/projects/data"
checkpoint="$data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_pair_at_then_av_fixed_band_window13_av_va_libritts50k_GRID_ctc003_warmup10k30k_40hz_char/model_100000.pt"
output_root="${OUTPUT_ROOT:-$data/evaluations/GRID_paired_at_then_av_window13_libritts_100k}"
gpus="${EVAL_GPUS:-0,1,2,3}"
workers="${WORKERS_PER_GPU:-4}"
mkdir -p "$output_root"
reuse=()
while read -r label setting text_mode; do
    output="$output_root/$label"
    mkdir -p "$output"
    "$python_bin" - "$output" "$project_root" <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess
import sys

output, project = map(Path, sys.argv[1:])
files = subprocess.check_output(["git", "ls-files", "-z", "--", "."], cwd=project).decode().split("\0")
records = []
for name in filter(None, files):
    path = project / name
    if path.suffix in {".py", ".yaml", ".yml", ".sh", ".toml", ".cfg"}:
        payload = path.read_bytes()
        records.append({"path": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
snapshot = output / "launch_source_snapshot.json"
if snapshot.exists():
    prior = json.loads(snapshot.read_text())
    if prior["project_dir"] != str(project) or prior["source_files"] != records:
        raise RuntimeError(f"Evaluation source changed since launch: {output}")
else:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    snapshot.write_text(json.dumps({"project_dir": str(project), "git_commit": revision,
                                    "source_files": records}, indent=2) + "\n")
PY
    printf '%s %s\n' "Starting GRID $label" "$output"
    if ! "$python_bin" -c 'import json,sys,pathlib; p=pathlib.Path(sys.argv[1]); sys.exit(not (p.is_file() and json.loads(p.read_text()).get("complete") is True))' "$output/pipeline_status.json"; then
        "$python_bin" -u -m aligndit.script.eval.run_grid_s1 \
            --checkpoint "$checkpoint" --step 100000 --setting "$setting" --text-conditioning "$text_mode" \
            --gpus "$gpus" --workers-per-gpu "$workers" --output-dir "$output" >> "$output/pipeline.log" 2>&1
    fi
    "$python_bin" -u scripts/verify_grid_evaluation.py --output-dir "$output" "${reuse[@]}" >> "$output/pipeline.log" 2>&1
    reuse=(--reuse-checkpoint-verification "$output/verification.json")
    printf '%s\n' "Verified GRID $label"
done <<'SETTINGS'
s1 s1 reference-target
s2_reference_target s2 reference-target
s2_target_only s2 target-only
SETTINGS
