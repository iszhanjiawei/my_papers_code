#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
source env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
TASK_ROOT=""
TASK_SMOKE=()
TASK_MODE=ema
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) TASK_ROOT="$2"; shift 2 ;;
    --smoke) TASK_SMOKE=(--smoke); TASK_MODE=smoke; shift ;;
    *) echo "Usage: $0 --root <runtime-directory> [--smoke]" >&2; exit 2 ;;
  esac
done
[[ -n "$TASK_ROOT" ]] || { echo "--root is required" >&2; exit 2; }
TASK_PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
TASK_CKPTS="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char"
mkdir -p "$TASK_ROOT/shared"
if [[ ! -f "$TASK_ROOT/shared/metadata.json" ]]; then
  CUDA_VISIBLE_DEVICES=0 "$TASK_PYTHON" -u -m aligndit.script.eval.prepare_grid_zeroshot \
    --shared-dir "$TASK_ROOT/shared" > "$TASK_ROOT/prepare.log" 2>&1
fi
TASK_PIDS=()
TASK_GPU=0
for TASK_STEP in 100000 150000; do
  TASK_OUT="$TASK_ROOT/${TASK_MODE}_${TASK_STEP}"
  mkdir -p "$TASK_OUT"
  setsid "$TASK_PYTHON" -u -m aligndit.script.eval.run_grid_zeroshot \
    --shared-dir "$TASK_ROOT/shared" --output-dir "$TASK_OUT" \
    --checkpoint "$TASK_CKPTS/model_${TASK_STEP}.pt" --step "$TASK_STEP" \
    --gpu "$TASK_GPU" --workers-per-gpu 4 "${TASK_SMOKE[@]}" \
    > "$TASK_OUT/pipeline.log" 2>&1 < /dev/null &
  TASK_PID=$!
  TASK_PIDS+=("$TASK_PID")
  echo "$TASK_PID" > "$TASK_OUT/pipeline.pid"
  echo "$TASK_MODE checkpoint=$TASK_STEP physical_gpu=$TASK_GPU pid=$TASK_PID output=$TASK_OUT"
  TASK_GPU=$((TASK_GPU+1))
done
TASK_STATUS=0
for TASK_PID in "${TASK_PIDS[@]}"; do
  wait "$TASK_PID" || TASK_STATUS=1
done
exit "$TASK_STATUS"
