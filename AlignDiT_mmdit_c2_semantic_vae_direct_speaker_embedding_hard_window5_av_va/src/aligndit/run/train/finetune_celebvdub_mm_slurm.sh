#!/usr/bin/env bash
# Historical launch entry retained only as source reference in this snapshot.
echo "This isolated snapshot trains only the five-frame hard AV+VA window." >&2
echo "Use src/aligndit/run/train/finetune_celebvdub_svae_speaker_hard_window5_av_va_4x4090.sh" >&2
exit 1

# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Slurm 提交专用：不设置 CUDA_VISIBLE_DEVICES（由 slurm 自动分配）
# 提交命令（在项目根目录执行）：
#   /share/slurm.pl --job-name=mmdit_prompt_stagewise --nodelist=xju-aslp8 --gpu 4 --num-threads 64 \
#       ./mmdit_prompt_stagewise.log bash src/aligndit/run/train/finetune_celebvdub_mm_slurm.sh
OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_SOCKET_IFNAME=ens5f0 \
NCCL_DEBUG=WARN \
PYTHONPATH=src \
${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --main_process_port 29556 \
    src/aligndit/script/train/finetune.py \
    --config-name finetune_celebvdub_mm
