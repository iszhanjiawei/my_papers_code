#!/usr/bin/env bash
# Foreground entry. For detached training invoke this script with setsid.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
if [[ ! ${ROOT_PREFIX+x} && -x /home/zjw524/ENTER/envs/aligndit/bin/python ]]; then
    export ROOT_PREFIX=/home
fi
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
# Four ranks use 1,000 epochs in the reference GRID schedule. Additional
# ranks consume more utterances per update, so scale the epoch budget with them.
if (( num_processes > 4 )); then
    export GRID_TRAIN_EPOCHS="${GRID_TRAIN_EPOCHS:-$((250 * num_processes))}"
fi
config_name="${GRID_CONFIG_NAME:-finetune_grid}"
if [[ ${1:-} == --config-name ]]; then
    config_name="${2:?--config-name requires a Hydra configuration name}"
    shift 2
fi
# Validate the exact overrides before starting TensorBoard or allocating DDP.
"$python_bin" -u src/aligndit/script/misc/check_grid_training.py \
    --config-name "$config_name" --num-processes "$num_processes" -- "$@"
bash scripts/start_grid_tensorboard.sh
distributed_args=()
if [[ "$num_processes" -gt 1 ]]; then
    distributed_args+=(--multi_gpu)
fi
exec "$python_bin" -u -m accelerate.commands.launch \
    "${distributed_args[@]}" \
    --mixed_precision bf16 --num_machines 1 --dynamo_backend no \
    --num_processes "$num_processes" --main_process_port "${TRAIN_PORT:-29640}" \
    src/aligndit/script/train/finetune_grid.py --config-name "$config_name" "$@"
