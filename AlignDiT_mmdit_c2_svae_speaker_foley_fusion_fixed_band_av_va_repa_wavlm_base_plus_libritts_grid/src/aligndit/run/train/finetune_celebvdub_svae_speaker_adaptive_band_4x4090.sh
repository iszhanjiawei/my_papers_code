#!/usr/bin/env bash
# Compatibility entry: use this snapshot's Foley AV/VA + REPA experiment.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/finetune_celebvdub_svae_speaker_foley_fixed_band_av_va_repa_wavlm_base_plus_4x4090.sh" "$@"
