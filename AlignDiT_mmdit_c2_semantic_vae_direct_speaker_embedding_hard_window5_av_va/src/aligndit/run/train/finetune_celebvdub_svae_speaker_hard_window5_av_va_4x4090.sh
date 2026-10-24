#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG=finetune_celebvdub_mm_c2_svae_speaker_hard_window5_av_va
export TRAIN_PORT="${TRAIN_PORT:-29631}"
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" "$@"
