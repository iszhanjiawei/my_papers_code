#!/bin/bash
set -euo pipefail

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

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
parent_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing AlignDiT Python interpreter: $python_bin" >&2
    exit 1
fi
if [[ ! -f "$parent_dir/model_70000.pt" || -L "$parent_dir/model_70000.pt" ]]; then
    echo "Missing regular S2c 70k checkpoint: $parent_dir/model_70000.pt" >&2
    exit 1
fi
if [[ ! -f "$parent_dir/training_contract.json" || -L "$parent_dir/training_contract.json" ]]; then
    echo "Missing regular S2c training contract: $parent_dir/training_contract.json" >&2
    exit 1
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
    echo "Semantic-VAE minimal fix requires four visible GPUs" >&2
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

max_updates=200000
run_until="${MINIMAL_FIX_RUN_UNTIL_UPDATE:-$max_updates}"
if [[ ! "$run_until" =~ ^[0-9]+$ || "$run_until" -le 0 || "$run_until" -gt "$max_updates" ]]; then
    echo "MINIMAL_FIX_RUN_UNTIL_UPDATE must be in [1, $max_updates], got $run_until" >&2
    exit 2
fi

echo "Launching one-stage Semantic-VAE minimal fix: 4x4090, BF16, exact stop=$run_until" >&2
exec env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    EXPECTED_WORLD_SIZE=4 \
    MINIMAL_FIX_RUN_UNTIL_UPDATE="$run_until" \
    OMP_NUM_THREADS=1 \
    NCCL_TIMEOUT=1200 \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_DEBUG=WARN \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    "$python_bin" -u -m accelerate.commands.launch \
        --multi_gpu \
        --mixed_precision bf16 \
        --num_processes 4 \
        --main_process_port 29575 \
        src/aligndit/script/train/finetune_semantic_vae_c2_minimal_fix.py \
        --config-name finetune_celebvdub_mm_c2_semantic_vae_minimal_fix
