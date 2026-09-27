#!/usr/bin/env bash
# This snapshot trains/evaluates local AV + blocked VA + WavLM REPA.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/start_av_local_no_va_repa_tensorboard.sh" "$@"
