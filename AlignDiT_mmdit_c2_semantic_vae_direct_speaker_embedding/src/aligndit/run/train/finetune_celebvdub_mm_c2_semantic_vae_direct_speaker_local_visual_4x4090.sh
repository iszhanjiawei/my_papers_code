#!/usr/bin/env bash
set -euo pipefail

# Reuse the validated four-GPU speaker launcher and replace its Hydra config.
# Hydra accepts a repeated --config-name; the final value takes precedence.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Audio-only blocks 12..17: gated visual-local attention, fixed Flowley window (radius 0.5 s)" >&2
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" \
    --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_ctc003_warmup \
    "$@"
