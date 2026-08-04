#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

GPU_IDS=${GPU_IDS:-2}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29621}
FRAME_BUDGET_PER_GPU=${FRAME_BUDGET_PER_GPU:?Set FRAME_BUDGET_PER_GPU for the memory benchmark}
MAX_SAMPLES=${MAX_SAMPLES:-32}

for value_name in NUM_PROCESSES FRAME_BUDGET_PER_GPU MAX_SAMPLES; do
    value=${!value_name}
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 1
    fi
done
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != NUM_PROCESSES )); then
    echo "GPU_IDS must contain exactly NUM_PROCESSES=${NUM_PROCESSES} entries, got: ${GPU_IDS}" >&2
    exit 1
fi
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "GPU_IDS must contain unique non-negative integers, got: ${GPU_IDS}" >&2
        exit 1
    fi
    SEEN_GPUS[${gpu_id}]=1
done

ACCELERATE_DISTRIBUTED_ARGS=(--num_machines 1)
if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_DISTRIBUTED_ARGS+=(--multi_gpu --gpu_ids all)
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1200}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/accelerate" launch \
    "${ACCELERATE_DISTRIBUTED_ARGS[@]}" \
    --mixed_precision bf16 \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    src/aligndit/script/misc/benchmark_semantic_vae_pretrain_memory.py \
    --frame-budget "${FRAME_BUDGET_PER_GPU}" \
    --max-samples "${MAX_SAMPLES}" \
    --expected-world-size "${NUM_PROCESSES}"
