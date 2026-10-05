#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
gpu_list="${TRAIN_GPUS:-0,1,2,3}"
IFS=',' read -ra gpu_ids <<< "$gpu_list"
[[ "${#gpu_ids[@]}" -gt 0 ]] || { echo 'TRAIN_GPUS must contain GPU indices' >&2; exit 1; }
for gpu in "${gpu_ids[@]}"; do
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "Invalid GPU index: $gpu" >&2; exit 1; }
    memory="$(nvidia-smi -i "$gpu" --query-gpu=memory.used --format=csv,noheader,nounits)"
    if ((memory > ${MAX_EXISTING_GPU_MEMORY_MB:-500})); then
        echo "GPU $gpu already uses $memory MiB; choose idle TRAIN_GPUS." >&2
        exit 1
    fi
done
mkdir -p logs
exec 9>logs/grid_training.lock
flock -n 9 || { echo 'A GRID training launcher is already active in this copy' >&2; exit 1; }
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}" NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# The persistent TensorBoard process must not inherit the training lock.
bash scripts/start_grid_tensorboard.sh 8>&- 9>&-
launch_args=(--num_processes "${#gpu_ids[@]}" --num_machines 1 --mixed_precision bf16 --dynamo_backend no)
if ((${#gpu_ids[@]} > 1)); then launch_args+=(--multi_gpu); fi
exec env CUDA_VISIBLE_DEVICES="$gpu_list" "$python_bin" -u -m accelerate.commands.launch \
    "${launch_args[@]}" --main_process_port "${TRAIN_PORT:-29735}" \
    src/aligndit/script/train/finetune_grid_semantic_vae.py --config-name finetune_grid_mmdit "$@"
