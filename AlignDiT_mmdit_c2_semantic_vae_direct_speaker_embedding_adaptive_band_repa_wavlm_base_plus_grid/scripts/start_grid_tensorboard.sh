#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
port="${TENSORBOARD_PORT:-6017}"
logdir="${TENSORBOARD_LOGDIR:-$project_root/runs}"
mkdir -p "$logdir" "$project_root/logs"
pidfile="$project_root/logs/tensorboard_grid_${port}.pid"
if [[ -f "$pidfile" ]]; then
    pid="$(cat "$pidfile")"
    if kill -0 "$pid" 2>/dev/null && curl --fail --silent "http://127.0.0.1:$port/data/logdir" | "$python_bin" -c 'import json,sys; from pathlib import Path; assert Path(json.load(sys.stdin)["logdir"]).resolve() == Path(sys.argv[1]).resolve()' "$logdir"; then
        printf 'TensorBoard PID=%s port=%s logdir=%s URL=http://127.0.0.1:%s\n' "$pid" "$port" "$logdir" "$port"
        exit 0
    fi
fi
if [[ -n "$(ss -ltnH "sport = :$port")" ]]; then
    echo "Port $port is occupied. Set TENSORBOARD_PORT to a free port." >&2
    exit 1
fi
setsid "$python_bin" -u -m tensorboard.main --logdir "$logdir" --host 0.0.0.0 --port "$port" \
    > "$project_root/logs/tensorboard_grid_${port}.log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
for ((attempt=0; attempt<30; attempt++)); do
    if curl --fail --silent "http://127.0.0.1:$port/data/logdir" > /dev/null; then
        printf 'TensorBoard PID=%s port=%s logdir=%s URL=http://127.0.0.1:%s\n' "$pid" "$port" "$logdir" "$port"
        exit 0
    fi
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
done
echo "TensorBoard did not start; inspect logs/tensorboard_grid_${port}.log" >&2
exit 1
