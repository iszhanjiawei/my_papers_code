#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
ATTEMPT_ID=${HUBERT40_ATTEMPT_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}

if [[ ! -x "${PYTHON}" ]]; then
    echo "AlignDiT Python is not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
echo "Starting HuBERT 40 Hz extraction: attempt=${ATTEMPT_ID}, nproc=${NPROC_PER_NODE}"

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    src/aligndit/script/misc/extract_librispeech_hubert40.py \
    --attempt-id "${ATTEMPT_ID}" \
    "$@"
