#!/bin/bash
# Crop mouth ROI from CelebV-Dub full-face videos.
# CPU-bound per video (~10s); we run multiple processes per GPU to use all CPU cores.
# NOTE: GPU 2 is used by another user (xwt523) -> we only use GPUs 0 1 3.
# Usage: bash crop_mouth_celebvdub.sh

GPUS=(0 1 3)              # GPUs to use (avoid GPU 2 = xwt523)
PROCS_PER_GPU=${1:-8}     # processes per GPU; 8x3=24 total (server-friendly)
SPLIT=${2:-train}

PYTHON=/home/zjw524/anaconda3/envs/aligndit/bin/python
SCRIPT=src/aligndit/script/misc/crop_mouth_celebvdub.py
INPUT_DIR=/home/zjw524/projects/data/CelebVDub/video/${SPLIT}
OUTPUT_DIR=/home/zjw524/projects/data/CelebVDub/video_mouth/${SPLIT}
MEAN_FACE=/home/zjw524/datasets/20words_mean_face.npy

NGPU=${#GPUS[@]}
NSHARD=$((NGPU * PROCS_PER_GPU))
echo "GPUs=${GPUS[*]}, procs/gpu=${PROCS_PER_GPU}, total shards=${NSHARD}, split=${SPLIT}"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

RANK=0
for ((p = 0; p < PROCS_PER_GPU; p++)); do
    for GPU in "${GPUS[@]}"; do
        OMP_NUM_THREADS=1 \
        CUDA_VISIBLE_DEVICES=${GPU} \
        PYTHONPATH=src \
        ${PYTHON} -u ${SCRIPT} \
            --input-dir ${INPUT_DIR} \
            --output-dir ${OUTPUT_DIR} \
            --mean-face ${MEAN_FACE} \
            --nshard ${NSHARD} \
            --rank ${RANK} \
            --device cuda \
        &
        RANK=$((RANK + 1))
    done
done
wait
echo "ALL CROP SHARDS DONE"
