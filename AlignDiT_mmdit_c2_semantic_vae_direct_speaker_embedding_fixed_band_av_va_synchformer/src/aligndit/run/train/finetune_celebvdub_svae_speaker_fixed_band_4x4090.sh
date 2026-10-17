#!/usr/bin/env bash
set -euo pipefail
echo "This isolated snapshot trains fixed AV+VA, not the copied AV-only experiment." >&2
echo "Use finetune_celebvdub_svae_speaker_fixed_band_av_va_4x4090.sh instead." >&2
echo "The legacy AV-only YAML is retained for regression tests only." >&2
exit 1
