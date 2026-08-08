#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
ATTEMPT_ID=${VIDEO40CACHE_ATTEMPT_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
MANIFEST_SHA256=${CELEBVDUB_SVAE_MANIFEST_SHA256:-a6478cce785748cbcefd87af54eafa9f654d735afa1c41b8f846e041cbc1286d}

if [[ ! -x "${PYTHON}" ]]; then
    echo "AlignDiT Python is not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ "${NPROC_PER_NODE}" -le 0 ]]; then
    echo "NPROC_PER_NODE must be positive" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    src/aligndit/script/misc/cache_celebvdub_video_40hz.py \
    --device cpu \
    --attempt-id "${ATTEMPT_ID}" \
    --expected-manifest-sha256 "${MANIFEST_SHA256}" \
    "$@"
