#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_CONFIG=finetune_celebvdub_mm_c2_svae_speaker_fixed_band_av_va
export TRAIN_PORT="${TRAIN_PORT:-29629}"
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" "$@"
