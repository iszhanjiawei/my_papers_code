#!/usr/bin/env bash
# This snapshot trains/evaluates local AV + blocked VA + WavLM REPA.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/finetune_celebvdub_svae_speaker_av_local_no_va_repa_wavlm_base_plus_4x4090.sh" "$@"
