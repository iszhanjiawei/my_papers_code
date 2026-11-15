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
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1200}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
num_processes="${NUM_PROCESSES:-$("$python_bin" -c 'import torch; print(torch.cuda.device_count())')}"
if [[ "$num_processes" != 4 ]]; then
    echo "Preserved CelebV-Dub recipe requires four workers; got: $num_processes" >&2
    exit 1
fi
# Preserve the source entry's occupancy guard for the selected physical GPUs.
"$python_bin" - <<'PYCHECK'
import os, subprocess
rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
used = {index.strip(): int(memory.strip()) for index, memory in (line.split(",") for line in rows.splitlines())}
selected = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
if len(selected) != 4 or len(set(selected)) != 4 or any(index not in used for index in selected):
    raise SystemExit("Set CUDA_VISIBLE_DEVICES to four distinct physical GPU indices")
for index in selected:
    if used[index] > 500:
        raise SystemExit(f"Refusing to start: GPU {index} already uses {used[index]} MiB")
PYCHECK
# Validate the exact overrides before starting TensorBoard or allocating DDP.
"$python_bin" -u scripts/check_celeb_repa_training.py \
    --num-processes "$num_processes" -- "$@"
bash scripts/start_celeb_repa_tensorboard.sh
distributed_args=()
if [[ "$num_processes" -gt 1 ]]; then
    distributed_args+=(--multi_gpu)
fi
exec "$python_bin" -u -m accelerate.commands.launch \
    "${distributed_args[@]}" \
    --mixed_precision bf16 --num_machines 1 --dynamo_backend no \
    --num_processes "$num_processes" --main_process_port "${TRAIN_PORT:-29648}" \
    src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker.py --config-name finetune_celebvdub_repa "$@"
