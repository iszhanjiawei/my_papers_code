#!/usr/bin/env bash
set -euo pipefail

# Run this script through setsid for a detached long-running training session.
# CUDA_VISIBLE_DEVICES, TPCA_MASTER_PORT and TPCA_CONFIG_NAME are overridable.
# Remaining arguments are passed to Hydra unchanged. Canary runs must override
# model.name so their events/checkpoints cannot resume or overwrite formal runs.
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
source "$experiment_root/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS=1
export NCCL_TIMEOUT=1200
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1
export PYTHONPATH="$experiment_root/src${PYTHONPATH:+:$PYTHONPATH}"

IFS=',' read -r -a tpca_devices <<< "$CUDA_VISIBLE_DEVICES"
tpca_num_processes="${TPCA_NUM_PROCESSES:-${#tpca_devices[@]}}"
if [[ ! "$tpca_num_processes" =~ ^[1-9][0-9]*$ ]]; then
    printf 'TPCA_NUM_PROCESSES must be a positive integer.\n' >&2
    exit 2
fi

tpca_launch=(
    "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u
    -m accelerate.commands.launch
    --num_processes "$tpca_num_processes" --num_machines 1
    --dynamo_backend no --mixed_precision bf16
    --main_process_port "${TPCA_MASTER_PORT:-29577}"
)
if (( tpca_num_processes > 1 )); then
    tpca_launch+=(--multi_gpu)
fi
tpca_launch+=(
    src/aligndit/script/train/finetune.py
    --config-name "${TPCA_CONFIG_NAME:-finetune_celebvdub_mm_d1_hunyuan_tpca}"
    "$@"
)

printf 'Experiment root: %s\n' "$experiment_root"
printf 'CUDA_VISIBLE_DEVICES: %s; processes: %s\n' "$CUDA_VISIBLE_DEVICES" "$tpca_num_processes"
if [[ "${TPCA_DRY_RUN:-0}" == 1 ]]; then
    printf 'Launch command: '
    printf '%q ' "${tpca_launch[@]}"
    printf '\n'
    exit 0
fi
exec "${tpca_launch[@]}"
