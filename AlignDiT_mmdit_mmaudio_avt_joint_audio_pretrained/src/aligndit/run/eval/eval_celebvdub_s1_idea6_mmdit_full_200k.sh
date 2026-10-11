#!/bin/bash
set -euo pipefail
# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Evaluate the scratch MMAudio-style AVT Joint DiT at 200k updates.

GEN_DIR="results/finetune_celebvdub_mmaudio_avt_joint_d1_scratch_200000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
PYTHON=${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python
WAVLM_CKPT=${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth
ASR_CKPT=${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3
EMO_CKPT=${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large
GT_AV_FEAT=data/CelebVDub/avhubert_feat
GPU_NUMS=1
CUDA_GPUS=${EVAL_GPU:-0}

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

echo "===== AVSync: Step1 extract generated-audio AV-HuBERT features ====="
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=${CUDA_GPUS} \
PYTHONPATH=src:${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq \
${PYTHON} -u src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 --rank 0 \
    --v-input-dir data/CelebVDub/video_mouth/test/test \
    --a-input-dir ${GEN_DIR}/test \
    --output-dir ${GEN_DIR}/avhubert_feat/test \
    --ckpt-path ${ROOT_PREFIX}/zjw524/projects/data/large_vox_iter5.pt \
    --user_dir ${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert

echo "===== AVSync: Step2 calculate AVSync ====="
CUDA_VISIBLE_DEVICES=${CUDA_GPUS} PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e avsync -g ${GEN_DIR} -n ${GPU_NUMS} \
    --gt_av_feat ${GT_AV_FEAT}

echo "===== ALL DONE ====="
