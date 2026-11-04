#!/usr/bin/env bash
# Dedicated entry for AVT joint + cosine 17-to-13-frame Gaussian AV/VA curriculum.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" "$@"
