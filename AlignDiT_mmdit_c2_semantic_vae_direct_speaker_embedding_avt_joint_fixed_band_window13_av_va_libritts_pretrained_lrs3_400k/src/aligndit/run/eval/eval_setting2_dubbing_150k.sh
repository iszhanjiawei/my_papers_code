#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != / && ! -f "$project_root/env.sh" ]]; do project_root="$(dirname "$project_root")"; done
source "$project_root/env.sh"
cd "$project_root"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_ctc003_warmup10k30k_40hz_CelebVDub_char"
output="${OUTPUT_DIR:-$checkpoint_dir/eval_s2_150000_gttext_original115_cfgtext5_cfgvideo2}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/run_setting2.py \
    --checkpoint "$checkpoint_dir/model_150000.pt" --output-dir "$output" \
    --gpu "${EVAL_GPU:-3}" --memory-fraction "${EVAL_MEMORY_FRACTION:-0.22}"
