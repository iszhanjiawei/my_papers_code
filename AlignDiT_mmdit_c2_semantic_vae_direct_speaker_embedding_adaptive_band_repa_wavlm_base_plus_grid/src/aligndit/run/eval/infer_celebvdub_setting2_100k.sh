#!/usr/bin/env bash
# Exact same Setting 2 inputs and sampler as 150k; independent 100k EMA outputs.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
source "$project_root/env.sh"
cd "$project_root"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
benchmark="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/Video-to-Speech-benchmark"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char"
output="$benchmark/results/Ours_100k"
PYTHONPATH="$project_root/src" "$python_bin" -u \
    -m aligndit.script.eval.infer_celebvdub_setting2 \
    --checkpoint "$checkpoint_dir/model_100000.pt" --step 100000 \
    --output-dir "$output" \
    --reference-cache "$benchmark/cache/ours_svae_campplus_setting2_sweep_v2" \
    --gpu "${EVAL_GPU:-0}" --seed 0 --nfe 32 --cfg-text 5 --cfg-video 2 --sway -1 "$@"
if [[ "${RUN_METRICS:-1}" == 1 ]]; then
    env -u PYTHONPATH "$python_bin" -u "$benchmark/scripts/run_setting2_evaluation.py" \
        --generated "$output" --with-emosim
fi
