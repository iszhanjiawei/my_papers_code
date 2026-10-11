#!/usr/bin/env bash
# Foreground entry. For detached training invoke this script with setsid.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
export PATH="$(dirname "$python_bin"):$PATH"
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
num_processes="${NUM_PROCESSES:-$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')}"
if [[ ! "$num_processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "No CUDA GPUs visible, or NUM_PROCESSES is invalid: $num_processes" >&2
    exit 1
fi
# Validate the exact overrides before starting TensorBoard or allocating DDP.
"$python_bin" -u src/aligndit/script/misc/check_grid_training.py \
    --num-processes "$num_processes" -- "$@"
bash scripts/start_grid_tensorboard.sh
distributed_args=()
if [[ "$num_processes" -gt 1 ]]; then
    distributed_args+=(--multi_gpu)
fi
exec "$python_bin" -u -m accelerate.commands.launch \
    "${distributed_args[@]}" \
    --mixed_precision bf16 --num_machines 1 --dynamo_backend no \
    --num_processes "$num_processes" --main_process_port "${TRAIN_PORT:-29640}" \
    src/aligndit/script/train/finetune_grid.py --config-name finetune_grid "$@"
