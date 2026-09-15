#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
run_name="AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_synchformer_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char"
logdir="${TENSORBOARD_LOGDIR:-$project_root/runs/$run_name}"
port="${TENSORBOARD_PORT:-6007}"
mkdir -p "$logdir" "$project_root/logs"
"$python_bin" - "$port" <<'PY'
import socket
import sys
with socket.socket() as s:
    try:
        s.bind(('0.0.0.0', int(sys.argv[1])))
    except OSError as error:
        raise SystemExit(f'TensorBoard port {sys.argv[1]} is unavailable; set TENSORBOARD_PORT: {error}')
PY
setsid "$python_bin" -u -m tensorboard.main \
    --logdir "$logdir" --host 0.0.0.0 --port "$port" \
    > "$project_root/logs/tensorboard_synchformer_${port}.log" 2>&1 < /dev/null &
tb_pid=$!
"$python_bin" - "$tb_pid" "$port" "$logdir" "$project_root" <<'PY'
import json
import os
from pathlib import Path
import sys
import time
import urllib.request
pid, port, logdir, project = sys.argv[1:]
url = f'http://127.0.0.1:{port}/'
for _ in range(50):
    try:
        os.kill(int(pid), 0)
        with urllib.request.urlopen(url, timeout=1) as response:
            if response.status == 200:
                break
    except (OSError, TimeoutError):
        time.sleep(1)
else:
    raise SystemExit(f'TensorBoard failed HTTP readiness: PID={pid}, port={port}; inspect logs')
record = {'pid': int(pid), 'port': int(port), 'logdir': logdir, 'local_url': url}
Path(project, 'logs', 'tensorboard_synchformer.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record, indent=2), flush=True)
PY
