#!/usr/bin/env bash
# Historical launch entry retained only as source reference in this snapshot.
echo "This isolated snapshot trains only the five-frame hard AV+VA window." >&2
echo "Use src/aligndit/run/train/finetune_celebvdub_svae_speaker_hard_window5_av_va_4x4090.sh" >&2
exit 1

OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
PYTHONPATH=src \
accelerate launch \
    --mixed_precision bf16 \
    --num_processes 8 \
    src/aligndit/script/train/pretrain.py \
    --config-name pretrain \
