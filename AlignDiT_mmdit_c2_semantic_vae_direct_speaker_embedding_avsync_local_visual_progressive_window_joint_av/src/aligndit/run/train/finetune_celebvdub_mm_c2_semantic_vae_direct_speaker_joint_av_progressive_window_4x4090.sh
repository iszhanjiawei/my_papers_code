#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_PORT="${TRAIN_PORT:-29621}"
echo "Joint A->V Flowley bias in blocks 0..11; retain gated visual tail 12..17; two independent fade schedules" >&2
exec bash "$script_dir/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh" \
    --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_joint_av_progressive_window_ctc003_warmup \
    "$@"
