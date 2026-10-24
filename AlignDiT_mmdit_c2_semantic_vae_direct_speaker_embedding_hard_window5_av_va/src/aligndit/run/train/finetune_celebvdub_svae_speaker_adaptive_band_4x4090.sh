#!/usr/bin/env bash
# Historical launch entry retained only as source reference in this snapshot.
echo "This isolated snapshot trains only the five-frame hard AV+VA window." >&2
echo "Use src/aligndit/run/train/finetune_celebvdub_svae_speaker_hard_window5_av_va_4x4090.sh" >&2
exit 1

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The copied launcher validates GPUs/parent artifacts and always imports this
# snapshot via PYTHONPATH=src. Override only the experiment configuration/port.
export TRAIN_CONFIG=finetune_celebvdub_mm_c2_svae_speaker_adaptive_band
export TRAIN_PORT="${TRAIN_PORT:-29624}"
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" "$@"
