#!/usr/bin/env bash
# Invoke under setsid for long unattended runs. This script owns only its workers.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != / && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
[[ -f "$project_root/env.sh" ]] || { echo "Cannot locate project env.sh" >&2; exit 1; }
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
entry="src/aligndit/script/misc/prepare_libritts_text_audio.py"
cache_root="${LIBRITTS_CACHE_ROOT:-${ROOT_PREFIX}/zjw524/projects/data/LibriTTS_text_audio_svae1000k_seed666_fp32}"
dataset_root="${LIBRITTS_DATASET_ROOT:-${ROOT_PREFIX}/zjw524/datasets/LibriTTS_460h/LibriTTS}"
speaker_checkpoint="${LIBRITTS_SPEAKER_CHECKPOINT:-${ROOT_PREFIX}/zjw524/projects/data/pretrained_models/campplus/campplus_cn_en_common.pt}"
codec_repo="${LIBRITTS_CODEC_REPO:-${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE}"
codec_checkpoint="${LIBRITTS_CODEC_CHECKPOINT:-${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k/dac/ema_state_dict.pth}"
log_dir="${LIBRITTS_PREPARE_LOG_DIR:-$cache_root/logs}"
stage="${1:-all}"
if [[ "$stage" != all && "$stage" != inventory && "$stage" != extract && "$stage" != audit ]]; then
    echo "Usage: $0 [all|inventory|extract|audit]; configure LIBRITTS_* and PREPARE_GPUS via environment" >&2
    exit 2
fi
[[ $# -le 1 ]] || { echo "Pass configuration through environment variables; unexpected extra arguments" >&2; exit 2; }
[[ -x "$python_bin" ]] || { echo "Missing Python interpreter: $python_bin" >&2; exit 1; }
export PYTHONPATH=src PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
mkdir -p "$log_dir"

if [[ "$stage" == inventory || ( "$stage" == all && ! -f "$cache_root/metadata.json" ) ]]; then
    "$python_bin" -u "$entry" inventory --cache-root "$cache_root" --dataset-root "$dataset_root" \
        --validation-speakers "${LIBRITTS_VALIDATION_SPEAKERS:-20}"
fi

if [[ "$stage" == extract || ( "$stage" == all && ! -f "$cache_root/complete.json" ) ]]; then
    [[ -f "$cache_root/metadata.json" ]] || { echo "Run inventory first: $cache_root" >&2; exit 1; }
    [[ -f "$speaker_checkpoint" ]] || { echo "Missing CAM++ checkpoint: $speaker_checkpoint" >&2; exit 1; }
    IFS=',' read -r -a prepare_gpus <<< "${PREPARE_GPUS:-0,1,2,3,4,5,6,7}"
    [[ "${#prepare_gpus[@]}" -gt 0 ]] || { echo "PREPARE_GPUS is empty" >&2; exit 1; }
    declare -A seen_gpus=()
    for gpu in "${prepare_gpus[@]}"; do
        [[ "$gpu" =~ ^[0-9]+$ && -z "${seen_gpus[$gpu]+x}" ]] || { echo "Invalid or duplicate GPU: $gpu" >&2; exit 1; }
        seen_gpus[$gpu]=1
    done
    worker_pids=()
    stop_workers() {
        if [[ "${#worker_pids[@]}" -gt 0 ]]; then
            kill "${worker_pids[@]}" 2>/dev/null || true
            wait || true
        fi
        exit 130
    }
    trap stop_workers INT TERM
    for rank in "${!prepare_gpus[@]}"; do
        gpu="${prepare_gpus[$rank]}"
        CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u "$entry" extract \
            --cache-root "$cache_root" --rank "$rank" --world-size "${#prepare_gpus[@]}" --device cuda:0 \
            --codec-repo "$codec_repo" --codec-checkpoint "$codec_checkpoint" \
            --speaker-checkpoint "$speaker_checkpoint" > "$log_dir/extract_rank_${rank}.log" 2>&1 &
        worker_pids+=("$!")
        echo "extract rank=$rank gpu=$gpu pid=${worker_pids[$rank]} log=$log_dir/extract_rank_${rank}.log"
    done
    failed=0
    for rank in "${!worker_pids[@]}"; do
        if wait "${worker_pids[$rank]}"; then
            echo "extract rank=$rank finished"
        else
            echo "extract rank=$rank failed; inspect $log_dir/extract_rank_${rank}.log" >&2
            failed=1
        fi
    done
    worker_pids=()
    trap - INT TERM
    [[ "$failed" -eq 0 ]] || { echo "Extraction incomplete; rerun extract to resume valid cached records" >&2; exit 1; }
fi

if [[ "$stage" == audit || "$stage" == all || "$stage" == extract ]]; then
    "$python_bin" -u "$entry" audit --cache-root "$cache_root"
fi
