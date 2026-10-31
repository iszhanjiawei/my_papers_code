# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Ablation: all-layers dual-stream MM-DiT (n_mm_layers = 18).
# NOTE: 当前 12+6 的训练占用 GPU 4,5,6,7。请先用 nvitop 确认下面这几张卡空闲，
#       且只操作 USER 为 zjw524 的进程；如需换卡，修改 CUDA_VISIBLE_DEVICES 即可。
CUDA_VISIBLE_DEVICES=4,5,6,7 \
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
    --config-name finetune_celebvdub_mm_full \
