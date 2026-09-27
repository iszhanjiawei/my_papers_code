#!/usr/bin/env bash
# This snapshot uses Visual Path Band + blocked VA + WavLM REPA.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/start_visual_path_band_no_va_repa_tensorboard.sh" "$@"
