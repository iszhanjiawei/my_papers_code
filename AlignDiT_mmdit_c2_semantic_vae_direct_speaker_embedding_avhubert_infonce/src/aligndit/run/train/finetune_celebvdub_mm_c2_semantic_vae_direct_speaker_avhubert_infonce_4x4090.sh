#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
if [[ ! -f "$project_root/env.sh" ]]; then
    echo "Cannot locate the isolated experiment env.sh" >&2
    exit 1
fi
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
parent_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech"
train_port="${TRAIN_PORT:-29640}"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing AlignDiT Python interpreter: $python_bin" >&2
    exit 1
fi
for parent_file in model_70000.pt training_contract.json; do
    if [[ ! -f "$parent_dir/$parent_file" || -L "$parent_dir/$parent_file" ]]; then
        echo "Missing regular S2c parent artifact: $parent_dir/$parent_file" >&2
        exit 1
    fi
done
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
    echo "This launcher requires four visible GPUs" >&2
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
if ss -ltnH "sport = :$train_port" | grep -q .; then
    echo "Training port $train_port is occupied; set TRAIN_PORT to an available port" >&2
    exit 1
fi

echo "Launching isolated AV-HuBERT InfoNCE: same S2c70k EMA, 4 GPUs, bf16, LR=5e-5, weight=0->0.05/10k, tau=0.07, gap=5" >&2
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
        --multi_gpu \
        --mixed_precision bf16 \
        --num_machines 1 \
        --dynamo_backend no \
        --num_processes 4 \
        --main_process_port "$train_port" \
        src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker_avhubert_infonce.py \
        --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_avhubert_infonce \
        "$@"
