#!/usr/bin/env bash
# Same-clip reference protocol for this isolated adaptive-band snapshot.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
if [[ ! -f "$project_root/env.sh" ]]; then
    echo "Cannot locate this experiment's env.sh" >&2
    exit 1
fi
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_ctc003_warmup10k30k_40hz_CelebVDub_char"
step="${CKPT_STEP:-200000}"
checkpoint="$checkpoint_dir/model_${step}.pt"
cfg_video="${CFG_VIDEO:-2.0}"
text_conditioning="${TEXT_CONDITIONING:-reference-target}"
text_suffix=""
case "$text_conditioning" in
    reference-target) ;;
    target-only) text_suffix="_target_only" ;;
    *) echo "Unsupported TEXT_CONDITIONING: $text_conditioning" >&2; exit 1 ;;
esac
output_region="${OUTPUT_REGION:-target}"
region_suffix=""
case "$output_region" in
    target) ;;
    raw-full) region_suffix="_raw_full" ;;
    *) echo "Unsupported OUTPUT_REGION: $output_region" >&2; exit 1 ;;
esac
output_dir="${OUTPUT_DIR:-$checkpoint_dir/eval_s1_${step}_adaptive_band_cfgv${cfg_video}${text_suffix}${region_suffix}}"
if [[ ! -f "$checkpoint" || -L "$checkpoint" ]]; then
    echo "Missing regular speaker-conditioned checkpoint: $checkpoint" >&2
    exit 1
fi
if [[ -d "$output_dir" ]] && find "$output_dir" -type f -print -quit | grep -q .; then
    echo "Refusing to overwrite non-empty inference directory: $output_dir" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" OMP_NUM_THREADS=1 PYTHONPATH="$project_root/src" \
"$python_bin" -u src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py \
    --checkpoint "$checkpoint" \
    --step "$step" \
    --config src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band.yaml \
    --output-dir "$output_dir" \
    --cfg-video "$cfg_video" \
    --cfg-text "${CFG_TEXT:-5.0}" \
    --text-conditioning "$text_conditioning" \
    --output-region "$output_region" \
    --seed "${INFER_SEED:-0}" \
    --nfe "${NFE:-32}" \
    --device cuda:0 \
    "$@"
