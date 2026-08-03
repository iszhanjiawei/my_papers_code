#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/venvs/semantic-vae/bin/python"
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
ATTEMPT_ID=${SVAECACHE_ATTEMPT_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}

if [[ ! -x "${PYTHON}" ]]; then
    echo "Semantic-VAE Python is not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ ! "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer, got: ${NPROC_PER_NODE}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
echo "Starting Semantic-VAE extraction: attempt=${ATTEMPT_ID}, nproc=${NPROC_PER_NODE}"

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    src/aligndit/script/misc/extract_librispeech_svae_latents.py \
    --attempt-id "${ATTEMPT_ID}" \
    "$@"
