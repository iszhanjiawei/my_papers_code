#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/pretrain_libritts_text_audio_8xa40.sh" \
  --config-name pretrain_libritts_text_audio_3090 "$@"
