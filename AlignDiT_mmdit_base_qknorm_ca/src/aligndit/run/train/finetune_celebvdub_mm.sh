CUDA_VISIBLE_DEVICES=4,5,6,7 \
OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_SOCKET_IFNAME=ens5f0 \
NCCL_DEBUG=WARN \
PYTHONPATH=src \
/home/zjw524/ENTER/envs/aligndit/bin/python -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --main_process_port 29556 \
    src/aligndit/script/train/finetune.py \
    --config-name finetune_celebvdub_mm \
