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
if ! [[ "$SYNC_WORKERS_PER_GPU" =~ ^[1-9][0-9]*$ ]]; then
  echo "SYNC_WORKERS_PER_GPU must be a positive integer" >&2
  exit 2
fi
# Publish/validate metadata before any worker starts. Workers only read it.
"$PYTHON_BIN" -u scripts/extract_synchformer.py --initialize-only "$@" \
  > "$SYNC_LOG_DIR/initialize.log" 2>&1
cat "$SYNC_LOG_DIR/initialize.log"
"$PYTHON_BIN" -u scripts/supervise_synchformer_workers.py \
  --gpus "$SYNC_GPUS" --workers-per-gpu "$SYNC_WORKERS_PER_GPU" \
  --batch-size "$SYNC_BATCH_SIZE" --num-threads "$SYNC_NUM_THREADS" \
  --log-dir "$SYNC_LOG_DIR" --max-worker-rss-mib "${SYNC_MAX_WORKER_RSS_MIB:-6144}" \
  -- "$@"
"$PYTHON_BIN" -u scripts/extract_synchformer.py --audit-only "$@" \
  > "$SYNC_LOG_DIR/coverage_audit.log" 2>&1
cat "$SYNC_LOG_DIR/coverage_audit.log"
