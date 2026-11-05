#!/usr/bin/env bash
# Train the original small-batch AT plan on eight 24 GiB RTX 3090 GPUs.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export CONFIG_NAME=pretrain_libritts_text_audio_3090_small_batch
exec bash "${SCRIPT_DIR}/pretrain_libritts_text_audio_8xa40.sh" "$@"
