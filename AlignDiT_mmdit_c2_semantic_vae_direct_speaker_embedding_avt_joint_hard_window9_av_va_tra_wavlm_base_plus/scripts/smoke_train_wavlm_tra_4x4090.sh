#!/usr/bin/env bash
# Three real-data optimizer updates at the production batch size/LR schedule.
# Writes only to the isolated smoke checkpoint and TensorBoard run.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
run_base="AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_ctc003_warmup10k30k"
exec bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_4x4090.sh \
  "model.name=${run_base}_smoke3_20261031" \
  "ckpts.save_dir=${ROOT_PREFIX}/zjw524/projects/data/ckpts/${run_base}_40hz_CelebVDub_char_smoke3_20261031" \
  optim.run_until_update=3 ckpts.last_per_updates=3 "$@"
