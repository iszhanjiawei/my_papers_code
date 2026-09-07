#!/usr/bin/env bash
# Start a persistent TPCA training session and its dedicated TensorBoard.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
source env.sh
mkdir -p logs
stamp="$(date +%Y%m%d_%H%M%S)"
log="logs/train_speaker_tpca_${stamp}.log"
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_tpca_4x4090.sh "$@" \
  > "$log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_tpca.pid
printf 'Train PID=%s log=%s\n' "$train_pid" "$project_root/$log"
bash scripts/start_speaker_tpca_tensorboard.sh
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
