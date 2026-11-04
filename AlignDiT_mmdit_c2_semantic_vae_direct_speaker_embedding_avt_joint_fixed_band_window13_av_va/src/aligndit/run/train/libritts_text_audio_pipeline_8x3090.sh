#!/usr/bin/env bash
# Run under setsid: preparation must finish and pass the full audit before training.
# Resume extraction by rerunning; resume training with ckpts.resume=/.../model_N.pt.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_DIR}"
source env.sh
export PYTHONUNBUFFERED=1
bash "${SCRIPT_DIR}/start_libritts_text_audio_tensorboard.sh"
child_pid=''
stop_child() {
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" || true
  fi
  exit 130
}
trap stop_child INT TERM
bash "${SCRIPT_DIR}/../misc/prepare_libritts_text_audio_8xa40.sh" all &
child_pid=$!
wait "${child_pid}"
child_pid=''
trap - INT TERM
printf '%s\n' 'Full LibriTTS cache audit passed; launching 8-GPU text/audio pretraining.'
exec bash "${SCRIPT_DIR}/pretrain_libritts_text_audio_8x3090.sh" "$@"
