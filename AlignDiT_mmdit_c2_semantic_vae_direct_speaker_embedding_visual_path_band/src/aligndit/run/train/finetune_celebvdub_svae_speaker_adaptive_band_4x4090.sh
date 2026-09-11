#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The copied launcher validates GPUs/parent artifacts and always imports this
# snapshot via PYTHONPATH=src. Override only the experiment configuration/port.
export TRAIN_CONFIG=finetune_celebvdub_mm_c2_svae_speaker_adaptive_band
export TRAIN_PORT="${TRAIN_PORT:-29624}"
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" "$@"
