#!/bin/bash
# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Evaluate idea6 MMDiT model at 50k steps on CelebV-Dub Setting 1
# CWD: ${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/alignDiT_baseline/AlignDiT

GEN_DIR="results/finetune_celebvdub_mm_50000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
PYTHON=${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python
WAVLM_CKPT=${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth
ASR_CKPT=${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3
EMO_CKPT=${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large
GT_AV_FEAT=data/CelebVDub/avhubert_feat
GPU_NUMS=4
CUDA_GPUS=4,5,6,7

echo "===== SPKSIM ====="
CUDA_VISIBLE_DEVICES=${CUDA_GPUS} PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e sim -g ${GEN_DIR} -n ${GPU_NUMS} \
    --wavlm_ckpt ${WAVLM_CKPT}

echo "===== WER ====="
CUDA_VISIBLE_DEVICES=${CUDA_GPUS} PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e wer -l en -g ${GEN_DIR} -n ${GPU_NUMS} \
    --asr_ckpt ${ASR_CKPT}

echo "===== EMOSIM ====="
CUDA_VISIBLE_DEVICES=${CUDA_GPUS} PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e emosim -g ${GEN_DIR} -n ${GPU_NUMS} \
    --emo_ckpt ${EMO_CKPT}

echo "===== AVSync: Step1 提取生成音频 AV-HuBERT 特征 ====="
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4 \
PYTHONPATH=src:${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq \
${PYTHON} -u src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 --rank 0 \
    --v-input-dir data/CelebVDub/video_mouth/test \
    --a-input-dir ${GEN_DIR}/test \
    --output-dir ${GEN_DIR}/avhubert_feat/test \
    --ckpt-path ${ROOT_PREFIX}/zjw524/projects/data/large_vox_iter5.pt \
    --user_dir ${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert

echo "===== AVSync: Step2 计算 AVSync 指标 ====="
CUDA_VISIBLE_DEVICES=${CUDA_GPUS} PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e avsync -g ${GEN_DIR} -n ${GPU_NUMS} \
    --gt_av_feat ${GT_AV_FEAT}

echo "===== ALL DONE ====="
