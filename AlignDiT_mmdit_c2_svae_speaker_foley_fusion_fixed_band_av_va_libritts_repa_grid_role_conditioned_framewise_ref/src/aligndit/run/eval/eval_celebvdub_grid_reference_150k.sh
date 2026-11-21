#!/usr/bin/env bash
# Frozen REPA lambda=0.1 EMA150k: one GRID reference for each of 213 targets.
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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}"
export PYTHONPATH="$project_root/src"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
data_root="${ROOT_PREFIX}/zjw524/projects/data"
checkpoint_dir="$data_root/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char"
grid_cache="${GRID_CACHE_DIR:-$data_root/Grid_reference_celebvdub_s1_seed0_svae1000k_campplus_v1}"
output_dir="${OUTPUT_DIR:-$checkpoint_dir/eval_gridref_150000_repa_baseplus_pairseed0_cfgv2.0}"
stage="${RUN_STAGE:-all}"
case "$stage" in all|prepare|infer|metrics) ;; *) echo "Invalid RUN_STAGE: $stage" >&2; exit 2 ;; esac

if [[ "$stage" == all || "$stage" == prepare ]]; then
    "$python_bin" -u -m aligndit.script.eval.prepare_grid_reference \
        --output-dir "$grid_cache" --pairing-seed 0 --device cuda:0
fi
if [[ "$stage" == all || "$stage" == infer ]]; then
    "$python_bin" -u -m aligndit.script.eval.infer_celebvdub_grid_reference \
        --pair-manifest "$grid_cache/pairs.jsonl" \
        --checkpoint "$checkpoint_dir/model_150000.pt" --step 150000 \
        --output-dir "$output_dir" --device cuda:0 \
        --seed 0 --nfe 32 --cfg-text 5.0 --cfg-video 2.0 --sway -1.0
fi
if [[ "$stage" == all || "$stage" == metrics ]]; then
    for task in sim wer emosim emoembed; do
        if [[ ! -f "$output_dir/_${task}_summary.json" ]]; then
            "$python_bin" -u -m aligndit.script.eval.eval_celebvdub_grid_reference \
                --manifest "$grid_cache/pairs.jsonl" --gen-wav-dir "$output_dir" -e "$task"
        fi
    done
    celebvdub="$data_root/CelebVDub"
    avhubert_fairseq="$data_root/av_hubert/fairseq/fairseq"
    if [[ ! -f "$output_dir/_avsync_summary.json" ]]; then
        PYTHONPATH="$project_root/src:$avhubert_fairseq" \
        "$python_bin" -u src/aligndit/script/misc/extract_avhubert.py \
            --nshard 1 --rank 0 \
            --v-input-dir "$celebvdub/video_mouth/test/test" \
            --a-input-dir "$output_dir/test" \
            --output-dir "$output_dir/avhubert_feat/test" \
            --ckpt-path "${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt" \
            --user_dir "$data_root/av_hubert/avhubert/avhubert"
        "$python_bin" -u -m aligndit.script.eval.eval_celebvdub_grid_reference \
            --manifest "$grid_cache/pairs.jsonl" --gen-wav-dir "$output_dir" -e avsync \
            --gt-av-feat "$celebvdub/avhubert_feat"
    fi
    "$python_bin" -u -m aligndit.script.eval.verify_grid_reference_results \
        --manifest "$grid_cache/pairs.jsonl" --output-dir "$output_dir"
    echo "GRID reference evaluation complete: $output_dir"
fi
