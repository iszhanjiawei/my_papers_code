#!/usr/bin/env bash
# This snapshot uses Visual Path Band + blocked VA + WavLM REPA.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/infer_celebvdub_s1_svae_speaker_visual_path_band_no_va_repa_wavlm_base_plus.sh" "$@"
