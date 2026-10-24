#!/usr/bin/env bash
# Historical launch entry retained only as source reference in this snapshot.
echo "This isolated snapshot trains only the five-frame hard AV+VA window." >&2
echo "Use src/aligndit/run/train/finetune_celebvdub_svae_speaker_hard_window5_av_va_4x4090.sh" >&2
exit 1

set -euo pipefail
echo "This isolated snapshot trains fixed AV+VA, not the copied AV-only experiment." >&2
echo "Use finetune_celebvdub_svae_speaker_fixed_band_av_va_4x4090.sh instead." >&2
echo "The legacy AV-only YAML is retained for regression tests only." >&2
exit 1
