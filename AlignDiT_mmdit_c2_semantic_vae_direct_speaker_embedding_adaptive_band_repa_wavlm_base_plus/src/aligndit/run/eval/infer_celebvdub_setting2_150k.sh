#!/usr/bin/env bash
# Reference-only VAE/CAM++ prompt + shared visual-only VSR, 115-sample Setting 2.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
source "$project_root/env.sh"
cd "$project_root"
export PYTHONPATH="$project_root/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char"
exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u     -m aligndit.script.eval.infer_celebvdub_setting2     --checkpoint "$checkpoint_dir/model_150000.pt" --step 150000     --gpu "${EVAL_GPU:-0}" --seed 0 --nfe 32 --cfg-text 5 --cfg-video 2 --sway -1 "$@"
