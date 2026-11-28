#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/infer_celebvdub_s1_svae_speaker_foley_fixed_band_av_va_repa_wavlm_base_plus.sh" "$@"
