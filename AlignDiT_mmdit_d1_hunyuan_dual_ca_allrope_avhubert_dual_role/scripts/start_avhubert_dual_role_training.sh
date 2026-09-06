#!/usr/bin/env bash
# Launch the independent D1 run and its TensorBoard service in separate sessions.
set -euo pipefail
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$experiment_root"
source "$experiment_root/env.sh"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
run_id="${D1_RUN_ID:-d1_avhubert_dual_role_$(date +%Y%m%d_%H%M%S)}"
export D1_MASTER_PORT="${D1_MASTER_PORT:-29586}"
tb_port="${D1_TENSORBOARD_PORT:-6008}"
model_name="AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_AVHuBERTDualRole_L6_W01_finetune"
tb_logdir="$experiment_root/runs/${model_name}_hifigan_16k_char_CelebVDub"
mkdir -p "$experiment_root/logs" "$tb_logdir"

"$python_bin" - "$D1_MASTER_PORT" "$tb_port" <<'PY'
import socket, sys
ports = [int(value) for value in sys.argv[1:]]
if len(set(ports)) != len(ports):
    raise SystemExit("Training and TensorBoard ports must differ")
for port in ports:
    with socket.socket() as sock:
        sock.bind(("0.0.0.0", port))
PY

setsid "$python_bin" -u -m tensorboard.main \
  --logdir "$tb_logdir" --host 0.0.0.0 --port "$tb_port" \
  > "$experiment_root/logs/${run_id}_tensorboard.log" 2>&1 < /dev/null &
tb_pid=$!
setsid env PYTHONUNBUFFERED=1 \
  bash "$experiment_root/src/aligndit/run/train/finetune_celebvdub_mm_d1_avhubert_dual_role_4x4090.sh" \
  > "$experiment_root/logs/${run_id}_train.log" 2>&1 < /dev/null &
train_pid=$!

"$python_bin" - "$experiment_root" "$run_id" "$train_pid" "$tb_pid" "$tb_port" "$tb_logdir" "$D1_MASTER_PORT" <<'PY'
import datetime, json, pathlib, sys
root, run, training_pid, tensorboard_pid, port, logdir, master_port = sys.argv[1:]
record = dict(
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    run_id=run, project_root=root, training_pid=int(training_pid),
    training_log=f"{root}/logs/{run}_train.log",
    tensorboard_pid=int(tensorboard_pid), tensorboard_port=int(port),
    tensorboard_logdir=logdir, tensorboard_log=f"{root}/logs/{run}_tensorboard.log",
    master_port=int(master_port), forwarded_address=None,
    config="finetune_celebvdub_mm_d1_hunyuan_dual_ca_allrope_avhubert_dual_role",
    status="launched_pending_worker_loss_and_http_verification",
)
path = pathlib.Path(root) / "logs" / f"{run}_launch.json"
path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
print(f"Launch record: {path}")
PY

ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid,$tb_pid"
