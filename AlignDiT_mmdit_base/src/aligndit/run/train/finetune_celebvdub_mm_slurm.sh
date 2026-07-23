# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Slurm 提交专用：不设置 CUDA_VISIBLE_DEVICES（由 slurm 自动分配）
# 提交命令（在项目根目录执行）：
#   sbatch --nodelist=xju-aslp8 -o sbatch_mmdit_base.log sbatch_train_mmdit_base.sh
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
    --main_process_port 29557 \
    src/aligndit/script/train/finetune.py \
    --config-name finetune_celebvdub_mm
