#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
if [[ ! -x "${PYTHON}" ]]; then
    echo "AlignDiT Python is not executable: ${PYTHON}" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

exec "${PYTHON}" -u src/aligndit/script/misc/prepare_celebvdub_svae_manifest.py "$@"
