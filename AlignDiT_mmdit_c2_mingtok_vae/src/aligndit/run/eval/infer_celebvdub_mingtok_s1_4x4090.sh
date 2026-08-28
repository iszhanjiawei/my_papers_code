#!/usr/bin/env bash
# CelebV-Dub Setting-1 inference for the strict 64D/50Hz MingTok C2 experiment.
# Usage: bash src/aligndit/run/eval/infer_celebvdub_mingtok_s1_4x4090.sh CHECKPOINT [OUTPUT_DIR]

set -euo pipefail

# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------

CHECKPOINT_PATH="${1:-${CKPT_PATH:-}}"
if [ -z "$CHECKPOINT_PATH" ]; then
    echo "Usage: $0 CHECKPOINT [OUTPUT_DIR]" >&2
    echo "CKPT_PATH may be used instead of the first argument." >&2
    exit 2
fi
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Checkpoint not found: $CHECKPOINT_PATH" >&2
    exit 2
fi

CHECKPOINT_NAME="$(basename "$CHECKPOINT_PATH")"
CHECKPOINT_STEM="${CHECKPOINT_NAME%.*}"
RESULT_DIR="${2:-${OUTPUT_DIR:-results/train_celebvdub_mingtok_c2_${CHECKPOINT_STEM}/celebvdub_test_s1}}"
CONFIG_PATH="${CONFIG_PATH:-src/aligndit/config/train_celebvdub_mingtok_c2.yaml}"
LOG_PATH="${LOG_PATH:-logs/infer_celebvdub_mingtok_s1_${CHECKPOINT_STEM}.log}"
PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"

mkdir -p "$(dirname "$LOG_PATH")"

INFER_ARGS=(
    --checkpoint "$CHECKPOINT_PATH"
    --config "$CONFIG_PATH"
    --output-dir "$RESULT_DIR"
    --seed "${SEED:-0}"
    --posterior-seed "${POSTERIOR_SEED:-666}"
    --nfe "${NFE:-32}"
    --ode-method "${ODE_METHOD:-euler}"
    --sway "${SWAY:-1.0}"
    --cfg-text "${CFG_TEXT:-5.0}"
    --cfg-video "${CFG_VIDEO:-2.0}"
    --codec-dtype "${MINGTOK_DTYPE:-bfloat16}"
    --codec-backend "${MINGTOK_BACKEND:-eager}"
)
if [ -n "${MAX_ITEMS:-}" ]; then
    INFER_ARGS+=(--max-items "$MAX_ITEMS")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
OMP_NUM_THREADS=1 \
NCCL_TIMEOUT=1200 \
NCCL_IB_DISABLE=1 \
NCCL_P2P_DISABLE=1 \
NCCL_DEBUG=WARN \
PYTHONUNBUFFERED=1 \
PYTHONPATH=src \
"$PYTHON_BIN" -u -m accelerate.commands.launch \
    --mixed_precision bf16 \
    --num_processes 4 \
    --main_process_port "${MAIN_PROCESS_PORT:-29574}" \
    src/aligndit/script/eval/infer_celebvdub_mingtok_s1.py \
    "${INFER_ARGS[@]}" \
    > "$LOG_PATH" 2>&1

echo "Inference complete: $RESULT_DIR"
echo "Log: $LOG_PATH"
