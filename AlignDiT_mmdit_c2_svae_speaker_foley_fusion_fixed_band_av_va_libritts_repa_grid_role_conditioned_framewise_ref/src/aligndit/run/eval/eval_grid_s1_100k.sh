#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$PROJECT_ROOT"
source env.sh
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m aligndit.script.eval.run_grid_s1 "$@"
