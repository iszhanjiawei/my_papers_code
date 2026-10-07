#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "6 MM + 12 audio: independent near-zero text and local visual gates in blocks 6..17" >&2
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" \
    --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_text_gate_ctc003_warmup \
    "$@"
