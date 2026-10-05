#!/usr/bin/env bash
# Compatibility entry in this isolated snapshot: always launch its own experiment.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$script_dir/finetune_celebvdub_mm_c2_framewise_visual_mod_4x4090.sh" "$@"
