#!/usr/bin/env bash
# Run under setsid. Set LIBRITTS_CACHE_ROOT / LIBRITTS_RUN_DIR for custom paths.
# WAIT_FOR_PREPARE_PID may attach to an already-running local preparation job.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_DIR}"
source env.sh
export PYTHONUNBUFFERED=1
export LIBRITTS_CACHE_ROOT="${LIBRITTS_CACHE_ROOT:-${ROOT_PREFIX}/zjw524/projects/data/LibriTTS_text_audio_svae1000k_seed666_fp32}"
RUN_DIR="${LIBRITTS_RUN_DIR:-${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_LibriTTS_text_audio_12AT6A_svae_speaker}"
for argument in "$@"; do
  if [[ "$argument" == ckpts.save_dir=* || "$argument" == datasets.cache_root=* ]]; then
    printf '%s\n' 'Use LIBRITTS_RUN_DIR / LIBRITTS_CACHE_ROOT so preparation and TensorBoard share the training paths.' >&2
    exit 2
  fi
done
if [[ -n "${LIBRITTS_RUN_DIR:-}" && -z "${TB_LOGDIR_SPEC:-}" ]]; then
  export TB_LOGDIR_SPEC="pretrain:${RUN_DIR}"
fi
bash "${SCRIPT_DIR}/start_libritts_text_audio_tensorboard.sh"
if [[ -n "${WAIT_FOR_PREPARE_PID:-}" ]]; then
  [[ "${WAIT_FOR_PREPARE_PID}" =~ ^[0-9]+$ ]] || exit 2
  printf 'Waiting for preparation PID %s and its complete audit.\n' "${WAIT_FOR_PREPARE_PID}"
  while kill -0 "${WAIT_FOR_PREPARE_PID}" 2>/dev/null; do
    process_state="$(ps -o stat= -p "${WAIT_FOR_PREPARE_PID}" || true)"
    [[ -n "${process_state}" && "${process_state}" != Z* ]] || break
    process_args="$(ps -o args= -p "${WAIT_FOR_PREPARE_PID}" || true)"
    [[ "${process_args}" == *prepare_libritts_text_audio_8xa40.sh* ]] || {
      printf '%s\n' 'Preparation PID no longer belongs to the expected launcher.' >&2
      exit 1
    }
    sleep 10
  done
  [[ -f "${LIBRITTS_CACHE_ROOT}/complete.json" ]] || {
    printf '%s\n' 'Preparation stopped without a complete audit; training was not started.' >&2
    exit 1
  }
else
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
  trap - INT TERM
fi
printf '%s\n' 'Preparation complete; the trainer will revalidate the data contract before starting.'
exec bash "${SCRIPT_DIR}/pretrain_libritts_text_audio_8xa40.sh" "$@" \
  "datasets.cache_root=${LIBRITTS_CACHE_ROOT}" "ckpts.save_dir=${RUN_DIR}"
