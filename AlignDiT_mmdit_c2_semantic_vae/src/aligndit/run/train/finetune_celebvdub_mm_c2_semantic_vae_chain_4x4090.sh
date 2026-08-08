#!/bin/bash
set -euo pipefail

start_stage="${1:-s3a}"
if [[ "$start_stage" != "s3a" && "$start_stage" != "s3b" ]]; then
    echo "Usage: $0 [s3a|s3b]" >&2
    exit 2
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
if [[ ! -f "$project_root/env.sh" ]]; then
    echo "Cannot locate project env.sh from $script_dir" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$project_root/env.sh"
cd "$project_root"

lock_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/.aligndit_semantic_vae_c2_s3_chain_4x4090.lock.d"
lock_owner="$lock_dir/owner.txt"
mkdir -p "$(dirname "$lock_dir")"
if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "Another Semantic-VAE C2 S3 chain owns $lock_dir" >&2
    if [[ -r "$lock_owner" ]]; then
        echo "Recorded owner:" >&2
        sed 's/^/  /' "$lock_owner" >&2
    fi
    exit 1
fi
cleanup_lock() {
    rm -f -- "$lock_owner" || true
    rmdir -- "$lock_dir" 2>/dev/null || true
}
trap cleanup_lock EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM
{
    printf 'host=%s\n' "$(hostname)"
    printf 'pid=%s\n' "$$"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'start_stage=%s\n' "$start_stage"
    printf 'project_root=%s\n' "$project_root"
} >"$lock_owner"

gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
if [[ "$gpu_count" -lt 4 ]]; then
    echo "Expected at least four GPUs, found $gpu_count" >&2
    exit 1
fi
while IFS=',' read -r gpu_index memory_used; do
    gpu_index="${gpu_index// /}"
    memory_used="${memory_used// /}"
    if [[ "$gpu_index" -lt 4 && "$memory_used" -gt 500 ]]; then
        echo "Refusing to start: GPU $gpu_index already uses ${memory_used} MiB" >&2
        exit 1
    fi
done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
stage_launcher="src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_4x4090.sh"
validator="src/aligndit/script/misc/validate_semantic_vae_c2_checkpoint.py"
logs_dir="$project_root/logs"
mkdir -p "$logs_dir"
run_stamp="$(date '+%Y%m%d_%H%M%S')"

s3a_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3a_40hz_CelebVDub_char"
s3b_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3b_40hz_CelebVDub_char"

run_stage() {
    local stage="$1"
    local stage_dir="$2"
    local log_path="$logs_dir/train_semantic_vae_c2_${stage}_4x4090_${run_stamp}.log"
    local started_at finished_at
    started_at="$(date --iso-8601=seconds)"
    echo "[$started_at] Starting $stage; log=$log_path" >&2
    S3_RUN_UNTIL_STAGE_UPDATE="$( [[ "$stage" == "s3a" ]] && echo 5000 || echo 195000 )" \
        bash "$stage_launcher" "$stage" >"$log_path" 2>&1
    env PYTHONPATH=src "$python_bin" -u "$validator" \
        --stage "$stage" \
        --checkpoint-dir "$stage_dir" >>"$log_path" 2>&1
    finished_at="$(date --iso-8601=seconds)"
    echo "[$finished_at] Completed and validated $stage" >&2
}

if [[ "$start_stage" == "s3a" ]]; then
    run_stage s3a "$s3a_dir"
else
    env PYTHONPATH=src "$python_bin" -u "$validator" --stage s3a --checkpoint-dir "$s3a_dir"
fi
run_stage s3b "$s3b_dir"
echo "Semantic-VAE C2 chain completed at cumulative update 200000" >&2
