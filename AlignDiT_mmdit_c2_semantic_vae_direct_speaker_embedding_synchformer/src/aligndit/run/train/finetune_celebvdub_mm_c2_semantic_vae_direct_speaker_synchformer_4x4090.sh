#!/usr/bin/env bash
# Same S2c 70k EMA parent, speaker/CTC settings, with frozen Synchformer video conditioning.
# Invoke with --check-only for CPU readiness validation without starting training.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
if [[ ! -f "$project_root/env.sh" ]]; then
    echo "Cannot locate this experiment's env.sh" >&2
    exit 1
fi
source "$project_root/env.sh"
cd "$project_root"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
export PYTHONPATH="$project_root/src"
export PYTHONUNBUFFERED=1
if [[ ! -x "$python_bin" ]]; then
    echo "Missing AlignDiT Python interpreter: $python_bin" >&2
    exit 1
fi
check_only=0
if [[ "${1:-}" == "--check-only" ]]; then
    check_only=1
    shift
fi
mkdir -p "$project_root/logs"
report="${PREFLIGHT_REPORT:-$project_root/logs/synchformer_preflight.json}"
"$python_bin" -u scripts/preflight_synchformer.py --report "$report" -- "$@"
if [[ "$check_only" == 1 ]]; then
    exit 0
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
num_gpus="${NUM_GPUS:-4}"
"$python_bin" - "$num_gpus" <<'PYGPU'
import sys
import torch
requested = int(sys.argv[1])
if requested < 1 or torch.cuda.device_count() != requested:
    raise SystemExit(f"NUM_GPUS={requested} does not match CUDA_VISIBLE_DEVICES ({torch.cuda.device_count()} devices)")
for i in range(requested):
    free, total = torch.cuda.mem_get_info(i)
    print(f"GPU {i}: {free / 2**30:.1f} / {total / 2**30:.1f} GiB free", flush=True)
PYGPU
# Start the loss dashboard only after all cache and checkpoint checks pass.
TENSORBOARD_LOGDIR="$("$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tensorboard_logdir"])' "$report")" \
    bash "$project_root/scripts/start_synchformer_tensorboard.sh"
echo "Launching isolated Direct-C2 + CAM++ + Synchformer; LR=5e-5, CTC 0@10k -> 0.03@30k" >&2
exec env \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    NCCL_TIMEOUT=1200 \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=1 \
    NCCL_DEBUG=WARN \
    "$python_bin" -u -m accelerate.commands.launch \
        --mixed_precision bf16 \
        --num_machines 1 \
        --dynamo_backend no \
        --num_processes "$num_gpus" \
        --main_process_port "${TRAIN_PORT:-29627}" \
        src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker_synchformer.py \
        --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_synchformer \
        "$@"
