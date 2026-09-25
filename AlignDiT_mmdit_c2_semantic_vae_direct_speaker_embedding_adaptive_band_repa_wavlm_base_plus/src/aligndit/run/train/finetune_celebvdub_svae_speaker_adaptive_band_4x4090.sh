#!/usr/bin/env bash
# Compatibility entry: use this snapshot's combined adaptive-band + REPA experiment.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/finetune_celebvdub_svae_speaker_adaptive_band_repa_wavlm_base_plus_4x4090.sh" "$@"
