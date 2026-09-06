#!/usr/bin/env bash
# Starts this experiment only. Both services have independent sessions (no nohup).
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
run_name="AlignDiT_MMDiT_D1_SemanticVAE_Original_CTC003_Fixed_semantic_vae_40hz_CelebVDub_char"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_D1_SemanticVAE_Original_CTC003_Fixed_40hz_CelebVDub_char"
tb_logdir="$project_root/runs/$run_name"
tb_port="${TB_PORT:-6006}"
ddp_port="${MAIN_PROCESS_PORT:-29593}"
if [[ -d "$checkpoint_dir" && "${RESUME:-0}" != 1 ]]; then
    echo "Existing run directory: $checkpoint_dir. Set RESUME=1 only to intentionally resume this exact run." >&2
    exit 1
fi
"$python_bin" - "$tb_port" "$ddp_port" <<'PY'
import socket
import sys
import tensorboard
ports = list(map(int, sys.argv[1:]))
if len(set(ports)) != 2:
    raise ValueError("TensorBoard and DDP require different ports")
for port in ports:
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", port))
PY
mkdir -p logs "$tb_logdir"
stamp="$(date +%Y%m%d_%H%M%S)"
tb_log="$project_root/logs/tensorboard_${stamp}.log"
train_log="$project_root/logs/train_${stamp}.log"

setsid "$python_bin" -u -m tensorboard.main \
    --logdir "$tb_logdir" --host 0.0.0.0 --port "$tb_port" \
    > "$tb_log" 2>&1 < /dev/null &
tb_pid=$!
setsid env PYTHONUNBUFFERED=1 MAIN_PROCESS_PORT="$ddp_port" \
    bash src/aligndit/run/train/finetune_celebvdub_mm_d1_semantic_vae_direct_4x4090.sh \
    > "$train_log" 2>&1 < /dev/null &
train_pid=$!

echo "Run: $run_name"
echo "Training PID: $train_pid"
echo "Training log: $train_log"
echo "Checkpoint directory: $checkpoint_dir"
echo "TensorBoard PID: $tb_pid"
echo "TensorBoard logdir: $tb_logdir"
echo "TensorBoard log: $tb_log"
echo "TensorBoard server-local URL: http://127.0.0.1:$tb_port"
echo "Open the client's Ports panel and use the actual forwarded address for port $tb_port."
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid,$tb_pid"
