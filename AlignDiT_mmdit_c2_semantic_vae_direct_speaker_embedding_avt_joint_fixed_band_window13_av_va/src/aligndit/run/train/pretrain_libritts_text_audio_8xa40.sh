#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${PROJECT_DIR}"

# Respect an explicitly exported prefix, including an explicitly empty value.
if [[ ! ${ROOT_PREFIX+x} ]]; then
  if [[ -x /home/zjw524/ENTER/envs/aligndit/bin/python ]]; then
    export ROOT_PREFIX=/home
  elif [[ -x /zjw524/ENTER/envs/aligndit/bin/python ]]; then
    export ROOT_PREFIX=""
  elif [[ -x /s7home/zjw524/ENTER/envs/aligndit/bin/python ]]; then
    export ROOT_PREFIX=/s7home
  else
    printf '%s\n' 'Set ROOT_PREFIX to the prefix containing zjw524/ENTER/envs/aligndit.' >&2
    exit 1
  fi
fi
source "${PROJECT_DIR}/env.sh"

PYTHON="${PYTHON:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
# The current A40 host stalls on CUDA peer transfers; follow the repository's
# existing single-host launchers. Override explicitly on verified P2P hardware.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29561}"
if [[ ! -x ${PYTHON} ]]; then
  printf 'Python environment not found: %s\n' "${PYTHON}" >&2
  exit 1
fi
launch=("${PYTHON}" -m accelerate.commands.launch --num_processes "${NUM_PROCESSES}"
        --num_machines 1 --mixed_precision bf16 --main_process_port "${MAIN_PROCESS_PORT}")
if (( NUM_PROCESSES > 1 )); then
  launch+=(--multi_gpu)
fi
exec "${launch[@]}" src/aligndit/script/train/pretrain_libritts_text_audio.py \
  --config-name "${CONFIG_NAME:-pretrain_libritts_text_audio_a40}" \
  "$@" "runtime.expected_world_size=${NUM_PROCESSES}"
