#!/bin/bash
set -euo pipefail

# D1 (6 MM + 12 audio blocks) on Semantic-VAE latents, CTC 0@10k -> 0.03@30k.
# Start long runs with setsid and provide TensorBoard as described in AGENTS.md.
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
    echo "Semantic-VAE D1 requires four visible GPUs" >&2
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

echo "Launching Semantic-VAE D1: LR=5e-5, CTC 0@10k -> 0.03@30k" >&2
exec env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    OMP_NUM_THREADS=1 \
    NCCL_TIMEOUT=1200 \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_DEBUG=WARN \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    "$python_bin" -u -m accelerate.commands.launch \
        --mixed_precision bf16 \
        --num_processes 4 \
        --main_process_port "${MAIN_PROCESS_PORT:-29587}" \
        src/aligndit/script/train/finetune_semantic_vae_d1_direct.py \
        --config-name finetune_celebvdub_mm_d1_semantic_vae_direct \
        "$@"
