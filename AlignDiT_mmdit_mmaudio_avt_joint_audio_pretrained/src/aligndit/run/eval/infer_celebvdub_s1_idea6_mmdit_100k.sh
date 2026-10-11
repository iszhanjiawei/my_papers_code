#!/bin/bash
set -euo pipefail
# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Inference for the scratch MMAudio-style AVT Joint DiT at 100k updates.

CKPT_PATH=${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMAudioMMDiT_AVTJoint_D1_6J12A_DualCTC6_12_Scratch_hifigan_16k_CelebVDub_char/model_100000.pt
CKPT_STEP=100000
EXP_NAME=finetune_celebvdub_mmaudio_avt_joint_d1_scratch
NFE=32
CFG_T=5
CFG_V=2
EVAL_GPU=${EVAL_GPU:-0}
VOCODER_PATH=${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000

OMP_NUM_THREADS=1 \
CUDA_VISIBLE_DEVICES=${EVAL_GPU} \
PYTHONPATH=src \
${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/eval/infer.py \
    -n ${EXP_NAME} \
    -s 0 \
    -t celebvdub_test_s1 \
    -nfe ${NFE} \
    -c ${CKPT_STEP} \
    --cfg_t ${CFG_T} \
    --cfg_v ${CFG_V} \
    --ckpt-path ${CKPT_PATH} \
    --vocoder-path ${VOCODER_PATH} \
    > logs/infer_celebvdub_s1_mmaudio_avt_scratch_ckpt${CKPT_STEP}.log 2>&1

echo "Inference done, log: logs/infer_celebvdub_s1_mmaudio_avt_scratch_ckpt${CKPT_STEP}.log"
