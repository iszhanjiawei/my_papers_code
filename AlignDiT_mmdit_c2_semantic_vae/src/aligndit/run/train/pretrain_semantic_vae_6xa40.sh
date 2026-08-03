#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

GPU_IDS=${GPU_IDS:-2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29620}
FRAME_BUDGET_PER_GPU=${FRAME_BUDGET_PER_GPU:-13500}

if [[ ! "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NUM_PROCESSES must be a positive integer, got: ${NUM_PROCESSES}" >&2
    exit 1
fi
if [[ ! "${FRAME_BUDGET_PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "FRAME_BUDGET_PER_GPU must be a positive integer, got: ${FRAME_BUDGET_PER_GPU}" >&2
    exit 1
fi
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

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/accelerate" launch \
    --mixed_precision bf16 \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    src/aligndit/script/train/pretrain_semantic_vae.py \
    --config-name pretrain_semantic_vae \
    datasets.batch_size_per_gpu="${FRAME_BUDGET_PER_GPU}"
