#!/bin/bash
set -euo pipefail

stage="${1:-}"
if [[ "$stage" != "s3a" && "$stage" != "s3b" ]]; then
    echo "Usage: $0 <s3a|s3b>" >&2
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

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing AlignDiT Python interpreter: $python_bin" >&2
    exit 1
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
    echo "Semantic-VAE C2 S3 requires four visible GPUs" >&2
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

config_name="finetune_celebvdub_mm_c2_semantic_vae_${stage}"
port=29571
expected_stage_updates=5000
hydra_overrides=()
if [[ "$stage" == "s3b" ]]; then
    port=29572
    expected_stage_updates=195000
    parent_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_s3a_40hz_CelebVDub_char"
    parent="$parent_dir/model_5000.pt"
    parent_contract="$parent_dir/training_contract.json"
    if [[ ! -f "$parent" || -L "$parent" || ! -f "$parent_contract" || -L "$parent_contract" ]]; then
        echo "S3b requires the validated S3a final checkpoint and contract in $parent_dir" >&2
        exit 1
    fi
    read -r parent_sha _ < <(sha256sum "$parent")
    read -r parent_contract_sha _ < <(sha256sum "$parent_contract")
    parent_size="$(stat -c '%s' "$parent")"
    hydra_overrides+=(
        "stage.expected_parent_sha256=$parent_sha"
        "stage.expected_parent_size=$parent_size"
        "stage.expected_parent_contract_sha256=$parent_contract_sha"
    )
fi

run_until="${S3_RUN_UNTIL_STAGE_UPDATE:-$expected_stage_updates}"
if [[ ! "$run_until" =~ ^[0-9]+$ || "$run_until" -le 0 || "$run_until" -gt "$expected_stage_updates" ]]; then
    echo "Invalid S3_RUN_UNTIL_STAGE_UPDATE=$run_until for $stage" >&2
    exit 2
fi

echo "Launching $stage on four RTX 4090 GPUs: local stop=$run_until, BF16, 3600 frames/GPU" >&2
exec env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    EXPECTED_WORLD_SIZE=4 \
    S3_RUN_UNTIL_STAGE_UPDATE="$run_until" \
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
        --main_process_port "$port" \
        src/aligndit/script/train/finetune_semantic_vae_c2.py \
        --config-name "$config_name" \
        "${hydra_overrides[@]}"
