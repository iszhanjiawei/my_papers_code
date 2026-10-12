#!/bin/bash
# 推理 + 评测: AlignDiT Semantic-VAE minimal-fix v2, step 150k, CelebV-Dub Setting-1
# CWD 必须为: /zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct

set -e
cd /zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct

PYTHON=/zjw524/ENTER/envs/aligndit/bin/python
CKPT=/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_40hz_CelebVDub_char/model_150000.pt
STEP=150000
OUTPUT_DIR=/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_minimal_fix_v2_40hz_CelebVDub_char/eval_s1_150000

CELEBVDUB=/zjw524/projects/alignDiT_idea6/papers_codes/alignDiT_baseline/AlignDiT/data/CelebVDub
GT_AV_FEAT=${CELEBVDUB}/avhubert_feat
TEST_LIST=/zjw524/projects/data/celebvdub_test_s1.lst
WAVLM=/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth
ASR=/zjw524/projects/data/faster-whisper-large-v3
EMO=/zjw524/projects/data/emotion2vec_plus_large
AVHUBERT_CKPT=/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt
AVHUBERT_USER_DIR=/zjw524/projects/data/av_hubert/avhubert/avhubert
AVHUBERT_FAIRSEQ=/zjw524/projects/data/av_hubert/fairseq/fairseq

mkdir -p logs

echo "===== [1/5] 推理生成音频 ====="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py \
    --checkpoint ${CKPT} \
    --step ${STEP} \
    --output-dir ${OUTPUT_DIR} \
    --test-list ${TEST_LIST} \
    --semantic-vae-repo /zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE \
    --semantic-vae-checkpoint /zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k \
    --device cuda:0

echo "===== [2/5] SPKSIM ====="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e sim -g ${OUTPUT_DIR} -n 1 \
    --test-list ${TEST_LIST} \
    --celebvdub-root ${CELEBVDUB} \
    --wavlm_ckpt ${WAVLM}

echo "===== [3/5] WER ====="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e wer -l en -g ${OUTPUT_DIR} -n 1 \
    --test-list ${TEST_LIST} \
    --celebvdub-root ${CELEBVDUB} \
    --asr_ckpt ${ASR}

echo "===== [4/5] EMOSIM ====="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e emosim -g ${OUTPUT_DIR} -n 1 \
    --test-list ${TEST_LIST} \
    --celebvdub-root ${CELEBVDUB} \
    --emo_ckpt ${EMO}

echo "===== [5/5] AVSync: 提取特征 ====="
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=${AVHUBERT_FAIRSEQ}:src \
${PYTHON} -u src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 --rank 0 \
    --v-input-dir ${CELEBVDUB}/video_mouth/test/test \
    --a-input-dir ${OUTPUT_DIR}/test \
    --output-dir ${OUTPUT_DIR}/avhubert_feat/test \
    --ckpt-path ${AVHUBERT_CKPT} \
    --user_dir ${AVHUBERT_USER_DIR}

echo "===== [5/5] AVSync: 计算指标 ====="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
${PYTHON} -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e avsync -g ${OUTPUT_DIR} -n 1 \
    --test-list ${TEST_LIST} \
    --celebvdub-root ${CELEBVDUB} \
    --gt_av_feat ${GT_AV_FEAT}

echo "===== ALL DONE: svae_minimal_fix_v2 @ step ${STEP} ====="
