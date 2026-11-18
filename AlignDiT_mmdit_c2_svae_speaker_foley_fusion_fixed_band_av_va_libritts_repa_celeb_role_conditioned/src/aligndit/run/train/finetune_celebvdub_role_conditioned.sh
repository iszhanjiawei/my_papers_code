#!/usr/bin/env bash
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
gpu_count="$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')"
num_processes="${NUM_PROCESSES:-$gpu_count}"
if [[ ! "$num_processes" =~ ^[1-9][0-9]*$ ]]; then
    echo "Invalid process count: $num_processes" >&2
    exit 1
fi
if [[ "${ALIGNDIT_SHARE_CUDA_DEVICE:-0}" == 1 ]]; then
    if (( gpu_count != 1 || num_processes < 2 )); then
        echo "Shared-GPU DDP requires one visible GPU and at least two workers" >&2
        exit 1
    fi
    # DDP uses each model's existing cuda:0 placement. Keep distinct ranks for data sharding.
    export ACCELERATE_BYPASS_DEVICE_MAP=true
    export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
    echo "Launching $num_processes training ranks on one GPU with Gloo gradient synchronization"
elif (( num_processes > gpu_count )); then
    echo "Requested $num_processes workers with $gpu_count visible GPUs" >&2
    exit 1
fi
"$python_bin" -u src/aligndit/script/misc/check_celebvdub_role_training.py --num-processes "$num_processes" -- "$@"
bash scripts/start_celeb_role_tensorboard.sh
distributed_args=()
if (( num_processes > 1 )); then
    distributed_args+=(--multi_gpu)
fi
exec "$python_bin" -u -m accelerate.commands.launch "${distributed_args[@]}" \
    --mixed_precision bf16 --num_machines 1 --dynamo_backend no \
    --num_processes "$num_processes" --main_process_port "${TRAIN_PORT:-29644}" \
    src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker.py \
    --config-name finetune_celebvdub_role_conditioned "$@"
