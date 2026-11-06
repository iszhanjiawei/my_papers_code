#!/usr/bin/env bash
# Fixed 13-frame experiment initialized from the requested LibriTTS 50k parent.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" "$@"
