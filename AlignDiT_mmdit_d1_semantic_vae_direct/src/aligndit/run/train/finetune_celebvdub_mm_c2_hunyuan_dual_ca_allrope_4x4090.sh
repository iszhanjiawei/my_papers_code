#!/usr/bin/env bash
set -euo pipefail

# C2 Hunyuan-style dual text CA on one 4 x RTX 4090 server. All training
# hyperparameters are inherited unchanged from the historical C2 config.
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
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

exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m accelerate.commands.launch \
    --multi_gpu --num_processes 4 --num_machines 1 --dynamo_backend no \
    --mixed_precision bf16 --main_process_port "${C2_MASTER_PORT:-29577}" \
    src/aligndit/script/train/finetune.py \
    --config-name finetune_celebvdub_mm_c2_hunyuan_dual_ca_allrope "$@"
