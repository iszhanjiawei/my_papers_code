#!/usr/bin/env bash
set -euo pipefail

__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"

EXP_NAME=finetune_celebvdub_mm_c2_campplus_speaker
MODEL_NAME=AlignDiT_MMDiT_qknorm_ca_c2_campplus_spk_tail6_finetune
CKPT_STEP=200000
CKPT_PATH="${ROOT_PREFIX}/zjw524/projects/data/ckpts/${MODEL_NAME}_hifigan_16k_CelebVDub_char/model_${CKPT_STEP}.pt"

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
PYTHONUNBUFFERED=1 \
PYTHONPATH=src \
"${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --main_process_port 29611 \
    src/aligndit/script/eval/infer.py \
    -n "$EXP_NAME" \
    -s 0 \
    -t celebvdub_test_s1 \
    -nfe 32 \
    -c "$CKPT_STEP" \
    --cfg_t 5 \
    --cfg_v 2 \
    --ckpt-path "$CKPT_PATH"
