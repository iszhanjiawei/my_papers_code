#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
port="${TENSORBOARD_PORT:-6012}"
logdir="$project_root/runs"
logfile="$project_root/logs/tensorboard_hard_window5_av_va_${port}.log"
mkdir -p "$logdir" "$project_root/logs"
if ss -ltnH "sport = :$port" | grep -q .; then
    echo "Port $port is already listening; choose TENSORBOARD_PORT explicitly" >&2
    exit 1
fi
setsid "$python_bin" -u -m tensorboard.main \
    --logdir "$logdir" --host 0.0.0.0 --port "$port" \
    > "$logfile" 2>&1 < /dev/null &
tb_pid=$!
"$python_bin" - "$port" "$tb_pid" "$logfile" <<'PY'
import os
import sys
import time
import urllib.request

port, pid, logfile = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
url = f"http://127.0.0.1:{port}/"
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
deadline = time.monotonic() + 45.0
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        raise SystemExit(f"TensorBoard exited before startup; inspect {logfile}")
    try:
        with opener.open(url, timeout=1) as response:
            if response.status == 200:
                print(f"TensorBoard HTTP 200: {url}", flush=True)
                break
    except OSError:
        pass
    time.sleep(0.2)
else:
    raise SystemExit(f"TensorBoard HTTP startup check failed; PID={pid}, log={logfile}")
PY
ss -ltnH "sport = :$port" | grep -q .
printf 'TensorBoard PID=%s port=%s logdir=%s\n' "$tb_pid" "$port" "$logdir"
ps -o pid,ppid,sid,tty,stat,cmd -p "$tb_pid"
