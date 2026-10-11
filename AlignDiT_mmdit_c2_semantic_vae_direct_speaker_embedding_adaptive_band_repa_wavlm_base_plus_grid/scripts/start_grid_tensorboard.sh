#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
logdir="$project_root/runs"
mkdir -p "$logdir" "$project_root/logs"
# A requested port must be free; otherwise select the first free port >=6020.
port="$("$python_bin" - <<'PY'
import os, socket
requested = os.environ.get('TENSORBOARD_PORT')
ports = [int(requested)] if requested else range(6020, 6120)
for port in ports:
    with socket.socket() as sock:
        try:
            sock.bind(('0.0.0.0', port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit('No free TensorBoard port; set TENSORBOARD_PORT to a free port')
PY
)"
setsid "$python_bin" -u -m tensorboard.main \
    --logdir "$logdir" --host 0.0.0.0 --port "$port" \
    > "$project_root/logs/tensorboard_grid_${port}.log" 2>&1 < /dev/null &
tb_pid=$!
"$python_bin" - "$tb_pid" "$port" "$logdir" "$project_root/logs/tensorboard_grid.json" <<'PY'
import json, os, sys, time, urllib.request
from pathlib import Path
pid, port, logdir, report = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
url = f'http://127.0.0.1:{port}/'
for _ in range(60):
    os.kill(pid, 0)
    try:
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(0.5)
else:
    raise SystemExit(f'TensorBoard HTTP startup failed; inspect logs/tensorboard_grid_{port}.log')
state = {'pid': pid, 'port': port, 'logdir': logdir, 'local_url': url}
report.write_text(json.dumps(state, indent=2) + '\n')
print(json.dumps(state, ensure_ascii=False))
print(f'打开底部“端口”页签，找到 {port}，点击该行的“转发地址”链接。')
PY
