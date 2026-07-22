#!/bin/bash
# Extract AV-HuBERT features (audio+video) for AVSync evaluation on CelebV-Dub test set.
# Usage: bash extract_avhubert_celebvdub_avsync.sh <GEN_WAV_DIR>

GEN_WAV_DIR=${1:-"results/finetune_celebvdub_200000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"}

PYTHON=/home/zjw524/ENTER/envs/aligndit/bin/python
FAIRSEQ_ROOT=/home/zjw524/projects/data/av_hubert/fairseq
FILE=src/aligndit/script/misc/extract_avhubert.py
CKPT=/home/zjw524/projects/data/large_vox_iter5.pt
USER_DIR=/home/zjw524/projects/data/av_hubert/avhubert

V_INPUT_DIR=data/CelebVDub/video_mouth/test
GT_A_INPUT_DIR=data/CelebVDub/audio/test
GT_OUTPUT_DIR=data/CelebVDub/avhubert_feat/test

GEN_A_INPUT_DIR=${GEN_WAV_DIR}/test
GEN_OUTPUT_DIR=${GEN_WAV_DIR}/avhubert_feat/test

echo "=== Step 1: Extract GT AV-HuBERT features ==="
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:${FAIRSEQ_ROOT} \
${PYTHON} -u ${FILE} \
    --nshard 1 \
    --rank 0 \
    --v-input-dir ${V_INPUT_DIR} \
    --a-input-dir ${GT_A_INPUT_DIR} \
    --output-dir ${GT_OUTPUT_DIR} \
    --ckpt-path ${CKPT} \
    --user_dir ${USER_DIR}

echo "=== Step 2: Extract Gen AV-HuBERT features ==="
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=src:${FAIRSEQ_ROOT} \
${PYTHON} -u ${FILE} \
    --nshard 1 \
    --rank 0 \
    --v-input-dir ${V_INPUT_DIR} \
    --a-input-dir ${GEN_A_INPUT_DIR} \
    --output-dir ${GEN_OUTPUT_DIR} \
    --ckpt-path ${CKPT} \
    --user_dir ${USER_DIR}

echo "=== Done! ==="
