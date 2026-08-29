#!/bin/bash
set -euo pipefail

# CelebVDub scratch training: 6 MMAudio-style AVT JointBlocks followed by
# 12 audio-only DiT blocks. No LibriSpeech AlignDiT checkpoint is loaded.

avt_envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$avt_envdir" != "/" ] && [ ! -f "$avt_envdir/env.sh" ]; do
    avt_envdir="$(dirname "$avt_envdir")"
done
if [ ! -f "$avt_envdir/env.sh" ]; then
    echo "Unable to locate experiment env.sh" >&2
    exit 1
fi
source "$avt_envdir/env.sh"
cd "$avt_envdir"

avt_visible_gpus="${AVT_VISIBLE_GPUS:-0,1,2,3}"
avt_main_port="${AVT_MAIN_PORT:-29571}"

CUDA_VISIBLE_DEVICES="$avt_visible_gpus" \
OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_DEBUG=WARN \
PYTHONUNBUFFERED=1 \
PYTHONPATH="$avt_envdir/src" \
"${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --main_process_port "$avt_main_port" \
    src/aligndit/script/train/finetune.py \
    --config-name finetune_celebvdub_mmaudio_avt_joint_d1_scratch \
    "$@"
