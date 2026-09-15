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
repa_cache="${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing AlignDiT Python interpreter: $python_bin" >&2
    exit 1
fi
for path in "$parent_dir/model_70000.pt" "$parent_dir/training_contract.json" \
    "$repa_cache/metadata.json" "$repa_cache/coverage_report.json"; do
    if [[ ! -f "$path" || -L "$path" ]]; then
        echo "Missing required regular file: $path" >&2
        exit 1
    fi
done
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
    echo "Direct-C2 REPA requires four visible GPUs" >&2
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

echo "Launching Direct-C2 + CAM++ + WavLM-Base+ REPA: tap=10th MM-DiT, lambda=0.05" >&2
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
        --num_machines 1 \
        --dynamo_backend no \
        --num_processes 4 \
        --main_process_port "${TRAIN_PORT:-29631}" \
        src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker.py \
        --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_repa_wavlm_base_plus \
        "$@"
