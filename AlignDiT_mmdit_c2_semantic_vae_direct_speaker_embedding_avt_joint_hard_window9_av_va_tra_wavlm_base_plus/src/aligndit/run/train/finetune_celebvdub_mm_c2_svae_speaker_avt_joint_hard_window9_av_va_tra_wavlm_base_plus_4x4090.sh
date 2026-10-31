#!/usr/bin/env bash
# Independent AVT hard-window9 + frozen WavLM timestep-adaptive alignment.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" "$@"
