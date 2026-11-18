#!/usr/bin/env bash
# Full original 115-pair text-condition ablation; generation followed by WER only.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
source "$project_root/env.sh"
cd "$project_root"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
benchmark="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/Video-to-Speech-benchmark"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char"
baseline="$benchmark/results/setting2_gttext_dubbing/Ours_150k"
output="${OUTPUT_DIR:-$benchmark/results/setting2_gttext_dubbing/Ours_150k_target_only}"
cache="${REFERENCE_CACHE:-$benchmark/cache/ours_svae_campplus_setting2_gttext_target_only}"
gpu="${EVAL_GPU:-0}"
lock="/tmp/alignDiT_idea6_vts_gpu${gpu}.lock"
PYTHONPATH="$project_root/src" "$python_bin" -u \
    -m aligndit.script.eval.infer_celebvdub_setting2_dubbing \
    --checkpoint "$checkpoint_dir/model_150000.pt" --step 150000 \
    --output-dir "$output" --reference-cache "$cache" --gpu "$gpu" --gpu-lock "$lock" \
    --seed 0 --nfe 32 --cfg-text 5 --cfg-video 2 --sway -1 \
    --text-input-mode target-only
if [[ ! -f "$output/setting2_wer.json" ]]; then
    flock "$lock" env -u PYTHONPATH CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u \
        "$benchmark/scripts/evaluate_setting2.py" --generated "$output" --metric wer
fi
PYTHONPATH="$project_root/src" "$python_bin" -u \
    -m aligndit.script.eval.verify_setting2_target_only_wer \
    --generated "$output" --baseline "$baseline" \
    --evaluator "$benchmark/scripts/evaluate_setting2.py"
