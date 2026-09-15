#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
source env.sh
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
SYNC_GPUS="${SYNC_GPUS:-0,1,2,3}"
SYNC_BATCH_SIZE="${SYNC_BATCH_SIZE:-8}"
SYNC_WORKERS_PER_GPU="${SYNC_WORKERS_PER_GPU:-4}"
SYNC_NUM_THREADS="${SYNC_NUM_THREADS:-2}"
SYNC_LOG_DIR="${SYNC_LOG_DIR:-$PROJECT_DIR/logs/synchformer_extraction}"
mkdir -p "$SYNC_LOG_DIR"
IFS=',' read -r -a GPU_IDS <<< "$SYNC_GPUS"
if ! [[ "$SYNC_WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNC_WORKERS_PER_GPU must be a positive integer" >&2
  exit 2
fi
WORKER_GPU_IDS=()
for gpu in "${GPU_IDS[@]}"; do
  for ((worker=0; worker<SYNC_WORKERS_PER_GPU; worker++)); do
    WORKER_GPU_IDS+=("$gpu")
  done
done
# Publish/validate metadata before any worker starts. Workers only read it.
"$PYTHON_BIN" -u scripts/extract_synchformer.py --initialize-only "$@" \
  > "$SYNC_LOG_DIR/initialize.log" 2>&1
cat "$SYNC_LOG_DIR/initialize.log"
PIDS=()
for rank in "${!WORKER_GPU_IDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${WORKER_GPU_IDS[$rank]}" "$PYTHON_BIN" -u scripts/extract_synchformer.py \
    --device cuda:0 --batch-size "$SYNC_BATCH_SIZE" --rank "$rank" --world-size "${#WORKER_GPU_IDS[@]}" --num-threads "$SYNC_NUM_THREADS" "$@" \
    > "$SYNC_LOG_DIR/rank${rank}.log" 2>&1 &
  PIDS+=("$!")
  echo "Synchformer rank=$rank gpu=${WORKER_GPU_IDS[$rank]} pid=${PIDS[-1]} log=$SYNC_LOG_DIR/rank${rank}.log"
done
failed=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
if [[ "$failed" -ne 0 ]]; then
  echo "Extraction worker failed; inspect rank logs and rerun to resume validated caches." >&2
  exit 1
fi
"$PYTHON_BIN" -u scripts/extract_synchformer.py --audit-only "$@" \
  > "$SYNC_LOG_DIR/coverage_audit.log" 2>&1
cat "$SYNC_LOG_DIR/coverage_audit.log"
