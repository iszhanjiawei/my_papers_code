#!/usr/bin/env bash
set -euo pipefail

# D1 VAE + frozen CAM++ speaker conditioning in all audio-only blocks 6..17.
# Use setsid for long runs and start TensorBoard as required by AGENTS.md.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../../.." && pwd)"
# shellcheck source=/dev/null
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
parent_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech"
speaker_cache="${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/campplus_spk_emb_zh_en_16k"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing AlignDiT Python interpreter: $python_bin" >&2
    exit 1
fi
for required_file in "$parent_dir/model_70000.pt" "$parent_dir/training_contract.json" \
    "$speaker_cache/metadata.json" "$speaker_cache/coverage_report.json"; do
    if [[ ! -f "$required_file" || -L "$required_file" ]]; then
        echo "Missing regular training resource: $required_file" >&2
        exit 1
    fi
done
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
    echo "D1 Semantic-VAE speaker training requires four GPUs" >&2
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

echo "Launching D1 Semantic-VAE + CAM++ audio-only tail12: CTC 0@10k -> 0.03@30k, stop at 200k" >&2
exec env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 \
    NCCL_TIMEOUT=1200 NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN \
    PYTHONUNBUFFERED=1 PYTHONPATH=src \
    "$python_bin" -u -m accelerate.commands.launch \
        --mixed_precision bf16 --num_machines 1 --dynamo_backend no --num_processes 4 \
        --main_process_port "${MAIN_PROCESS_PORT:-29588}" \
        src/aligndit/script/train/finetune_semantic_vae_d1_direct_speaker.py \
        --config-name finetune_celebvdub_mm_d1_semantic_vae_direct_speaker_ctc003_warmup \
        "$@"
