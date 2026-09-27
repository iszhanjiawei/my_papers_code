#!/bin/bash
# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Inference for idea6 MMDiT model at 50k steps on CelebV-Dub Setting 1
# CWD: ${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/alignDiT_baseline/AlignDiT

CKPT_PATH=${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_prompt_isolated_stagewise_finetune_hifigan_16k_CelebVDub_char/model_100000.pt
CKPT_STEP=100000
EXP_NAME=finetune_celebvdub_mm
NFE=32
CFG_T=5
CFG_V=2

OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=3600 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_SOCKET_IFNAME=ens5f0 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTHONPATH=src \
${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --main_process_port 29591 \
    src/aligndit/script/eval/infer.py \
    -n ${EXP_NAME} \
    -s 0 \
    -t celebvdub_test_s1 \
    -nfe ${NFE} \
    -c ${CKPT_STEP} \
    --cfg_t ${CFG_T} \
    --cfg_v ${CFG_V} \
    --ckpt-path ${CKPT_PATH} \
    > logs/infer_celebvdub_s1_idea6_mmdit_ckpt${CKPT_STEP}.log 2>&1

echo "Inference done, log: logs/infer_celebvdub_s1_idea6_mmdit_ckpt${CKPT_STEP}.log"
