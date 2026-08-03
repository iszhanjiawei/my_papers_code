#!/bin/bash
# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Extract AV-HuBERT features from cropped mouth videos for CelebV-Dub.
# Usage: bash extract_avhubert_celebvdub.sh [NGPU]

# NOTE: GPU 2 is used by another user (xwt523) -> only use GPUs 0 1 3.
GPUS=(0 1 3)
PROCS_PER_GPU=${1:-4}     # AV-HuBERT Large is GPU-heavy; 4x3=12 total
SPLIT=${2:-train}

PYTHON=${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python
FILE=src/aligndit/script/misc/extract_avhubert_from_only_video.py
INPUT_DIR=${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/video_mouth/${SPLIT}
OUTPUT_DIR=${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/avhubert_video_feat/${SPLIT}
CKPT=${ROOT_PREFIX}/zjw524/datasets/large_vox_iter5.pt
USER_DIR=${ROOT_PREFIX}/zjw524/projects/av_hubert/avhubert

NGPU=${#GPUS[@]}
NSHARD=$((NGPU * PROCS_PER_GPU))
echo "GPUs=${GPUS[*]}, procs/gpu=${PROCS_PER_GPU}, total shards=${NSHARD}, split=${SPLIT}"

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

RANK=0
for ((p = 0; p < PROCS_PER_GPU; p++)); do
    for gpu in "${GPUS[@]}"; do
        CUDA_VISIBLE_DEVICES=${gpu} \
        PYTHONPATH=src \
        ${PYTHON} -u ${FILE} \
            --nshard ${NSHARD} \
            --rank ${RANK} \
            --input-dir ${INPUT_DIR} \
            --output-dir ${OUTPUT_DIR} \
            --ckpt-path ${CKPT} \
            --user_dir ${USER_DIR} \
            --file-extension .mp4 \
        &
        RANK=$((RANK + 1))
    done
done
wait
echo "ALL AVHUBERT SHARDS DONE"
