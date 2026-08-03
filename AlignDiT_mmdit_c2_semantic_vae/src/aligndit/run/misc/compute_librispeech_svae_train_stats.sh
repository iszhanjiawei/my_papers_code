#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

cd "${PROJECT_ROOT}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
    src/aligndit/script/misc/compute_librispeech_svae_train_stats.py \
    "$@"
