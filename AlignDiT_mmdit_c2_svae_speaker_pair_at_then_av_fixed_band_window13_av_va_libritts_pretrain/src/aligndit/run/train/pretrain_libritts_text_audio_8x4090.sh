#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$project_root"
source env.sh
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m accelerate.commands.launch \
    --multi_gpu --num_processes 8 --num_machines 1 --dynamo_backend no \
    --mixed_precision bf16 --main_process_port "${TRAIN_PORT:-29664}" \
    src/aligndit/script/train/pretrain_libritts_text_audio.py \
    --config-name pretrain_libritts_text_audio "$@"
