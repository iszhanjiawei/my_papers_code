#!/bin/bash
# D0 on one 4×RTX 4090 server: six MM-DiT blocks, twelve text-free audio blocks,
# and one CTC head after the 12th block.

# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------

CUDA_VISIBLE_DEVICES=0,1,2,3 \
OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_DEBUG=WARN \
PYTHONUNBUFFERED=1 \
PYTHONPATH=src \
${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --main_process_port 29564 \
    src/aligndit/script/train/finetune.py \
    --config-name finetune_celebvdub_mm_d0_6mm12audio_ctc12
