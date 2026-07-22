#!/bin/bash
# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Inference for CelebV-Dub Setting 1 (GT speech as reference)
# Usage: bash src/aligndit/run/eval/infer_celebvdub_s1.sh
# CWD: AlignDiT project root

CKPT_STEP=200000
EXP_NAME=finetune_celebvdub
NFE=32
CFG_T=5
CFG_V=2

OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=3600 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_SOCKET_IFNAME=ens5f0 \
PYTHONPATH=src \
nohup ${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    src/aligndit/script/eval/infer.py \
    -n ${EXP_NAME} \
    -s 0 \
    -t celebvdub_test_s1 \
    -nfe ${NFE} \
    -c ${CKPT_STEP} \
    --cfg_t ${CFG_T} \
    --cfg_v ${CFG_V} \
    > logs/infer_celebvdub_s1_ckpt${CKPT_STEP}.log 2>&1 &

echo "Inference started, PID=$!, log: logs/infer_celebvdub_s1_ckpt${CKPT_STEP}.log"
