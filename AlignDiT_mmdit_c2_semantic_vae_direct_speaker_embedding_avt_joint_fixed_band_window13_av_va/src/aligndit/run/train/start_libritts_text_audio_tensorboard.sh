#!/usr/bin/env bash
# Start a detached TensorBoard for this experiment; reuse only a matching service.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
source "${PROJECT_DIR}/env.sh"
PYTHON="${PYTHON:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
RUNTIME_DIR="${RUNTIME_DIR:-${ROOT_PREFIX}/zjw524/projects/data/LibriTTS_text_audio_runtime}"
CKPT_ROOT="${ROOT_PREFIX}/zjw524/projects/data/ckpts"
export TB_PORT="${TB_PORT:-6026}"
export TB_LOGDIR_SPEC="${TB_LOGDIR_SPEC:-smoke:${CKPT_ROOT}/LibriTTS_text_audio_smoke,pretrain:${CKPT_ROOT}/AlignDiT_LibriTTS_text_audio_12AT6A_svae_speaker,memory:${RUNTIME_DIR}/memory_check}"
mkdir -p "${RUNTIME_DIR}"
service_matches() {
  "${PYTHON}" - <<'PY'
import json, os, urllib.request
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open('http://127.0.0.1:' + os.environ['TB_PORT'] + '/data/environment', timeout=2) as response:
        data = json.load(response)
    raise SystemExit(0 if data['data_location'] == os.environ['TB_LOGDIR_SPEC'] else 1)
except (OSError, KeyError, ValueError):
    raise SystemExit(1)
PY
}
if ! service_matches; then
  "${PYTHON}" - <<'PY'
import os, socket
with socket.socket() as sock:
    sock.bind(('0.0.0.0', int(os.environ['TB_PORT'])))
PY
  setsid "${PYTHON}" -u -m tensorboard.main --logdir_spec "${TB_LOGDIR_SPEC}" \
    --host 0.0.0.0 --port "${TB_PORT}" \
    > "${RUNTIME_DIR}/tensorboard_${TB_PORT}.log" 2>&1 < /dev/null &
  printf '%s\n' "$!" > "${RUNTIME_DIR}/tensorboard_$(hostname)_${TB_PORT}.pid"
fi
for ((attempt=0; attempt<15; attempt++)); do
  if service_matches; then
    printf 'TensorBoard port=%s logdir_spec=%s\n' "${TB_PORT}" "${TB_LOGDIR_SPEC}"
    pid_file="${RUNTIME_DIR}/tensorboard_$(hostname)_${TB_PORT}.pid"
    if [[ -f "${pid_file}" ]]; then
      ps -o pid,ppid,sid,tty,stat,args -p "$(cat "${pid_file}")"
    fi
    exit 0
  fi
  sleep 1
done
printf 'TensorBoard did not become ready; inspect %s\n' "${RUNTIME_DIR}/tensorboard_${TB_PORT}.log" >&2
exit 1
