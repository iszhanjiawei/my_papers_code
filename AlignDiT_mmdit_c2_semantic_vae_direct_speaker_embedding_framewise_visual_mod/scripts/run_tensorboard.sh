#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(dirname "$script_dir")"
# shellcheck source=/dev/null
source "$project_root/env.sh"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
logdir="${TENSORBOARD_LOGDIR:-$project_root/runs}"
port="${TENSORBOARD_PORT:-6006}"
mkdir -p "$logdir"
exec "$python_bin" -u -m tensorboard.main \
    --logdir "$logdir" --host 0.0.0.0 --port "$port" \
    --reload_interval 5 --load_fast false
