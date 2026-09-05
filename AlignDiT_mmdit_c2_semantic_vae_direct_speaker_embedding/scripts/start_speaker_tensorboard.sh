#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
port="${TENSORBOARD_PORT:-6006}"
run_name="AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char"
logdir="${TENSORBOARD_LOGDIR:-$project_root/runs}"
mkdir -p "$logdir" "$project_root/logs"
if ss -ltnH "sport = :$port" | grep -q .; then
    echo "Port $port is already listening; choose TENSORBOARD_PORT explicitly" >&2
    exit 1
fi
setsid "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m tensorboard.main \
    --logdir "$logdir" --host 0.0.0.0 --port "$port" \
    > "$project_root/logs/tensorboard_speaker_${port}.log" 2>&1 < /dev/null &
tb_pid=$!
printf 'TensorBoard PID=%s port=%s logdir=%s\n' "$tb_pid" "$port" "$logdir"
