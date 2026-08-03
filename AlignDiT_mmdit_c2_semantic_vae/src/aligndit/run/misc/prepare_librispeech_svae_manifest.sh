#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
cd "${PROJECT_ROOT}"

PYTHONPATH=src "${PYTHON}" -u src/aligndit/script/misc/prepare_librispeech_svae_manifest.py "$@"
