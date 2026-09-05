#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$project_root/env.sh"
cd "$project_root"
tensorboard_port="${TENSORBOARD_PORT:-6007}"
if [[ ! "$tensorboard_port" =~ ^[0-9]+$ ]] || (( tensorboard_port < 1 || tensorboard_port > 65535 )); then
    echo "Invalid TENSORBOARD_PORT: $tensorboard_port" >&2
    exit 1
fi
tensorboard_run="AlignDiT_MMDiT_qknorm_ca_d1_semantic_vae_direct_speaker_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char"
tensorboard_logdir="${TENSORBOARD_LOGDIR:-$project_root/runs/$tensorboard_run}"
if ss -ltnH "sport = :$tensorboard_port" | grep -q .; then
    echo "Port $tensorboard_port is already listening; choose TENSORBOARD_PORT explicitly" >&2
    exit 1
fi
mkdir -p "$tensorboard_logdir" "$project_root/logs"
tensorboard_stamp="$(date +%Y%m%d_%H%M%S)"
tensorboard_log="$project_root/logs/tensorboard_speaker_${tensorboard_port}_${tensorboard_stamp}.log"
set -o noclobber
setsid "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m tensorboard.main \
    --logdir "$tensorboard_logdir" --host 0.0.0.0 --port "$tensorboard_port" --reload_interval 5 \
    > "$tensorboard_log" 2>&1 < /dev/null &
tensorboard_pid=$!
printf 'TB_PID=%s\nTB_PORT=%s\nTB_LOGDIR=%s\nTB_LOG=%s\n' \
    "$tensorboard_pid" "$tensorboard_port" "$tensorboard_logdir" "$tensorboard_log"
# This prints launch metadata only. Verify process, HTTP, scalar updates and
# the actual client port-forwarding URL before declaring startup complete.
