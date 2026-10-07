#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != / && ! -f "$project_root/env.sh" ]]; do project_root="$(dirname "$project_root")"; done
[[ -f "$project_root/env.sh" ]] || { echo 'Cannot locate project env.sh' >&2; exit 1; }
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_GRID_svae_speaker_adaptive_band_repa_wavlm_base_plus_100k"
checkpoint="${CHECKPOINT:-$checkpoint_dir/model_100000.pt}"
output="${OUTPUT_DIR:-$checkpoint_dir/eval_grid_setting2_100000_seed666_cfg5_2}"
manifest="${GRID_EVAL_MANIFEST:-${ROOT_PREFIX}/zjw524/projects/grid_eval_20261001/assets/manifest.jsonl}"
assets="$(dirname "$manifest")"
generation_shards="${GENERATION_SHARDS:-8}"
wer_shards="${WER_SHARDS:-5}"
setting="${GRID_SETTING:-2}"
mkdir -p "$output/logs"
exec 9>"$output/evaluation.lock"
flock -n 9 || { echo "Evaluation is already active: $output" >&2; exit 1; }

if [[ -f "$output/metrics_summary.json" ]]; then
    echo "Evaluation already complete: $output/metrics_summary.json"
    cat "$output/metrics_summary.json"
    exit 0
fi
if [[ "$generation_shards" -gt 8 || "$generation_shards" -lt 1 ]]; then
    echo 'GENERATION_SHARDS must be between 1 and 8' >&2
    exit 1
fi
if [[ "$wer_shards" -gt 5 || "$wer_shards" -lt 1 ]]; then
    echo 'WER_SHARDS must be between 1 and 5' >&2
    exit 1
fi

export PYTHONPATH="$project_root/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

wait_group() {
    local label=$1 status=0
    shift
    for pid in "$@"; do wait "$pid" || status=1; done
    if [[ "$status" -ne 0 ]]; then
        echo "$label failed; inspect $output/logs" >&2
        exit 1
    fi
}

echo "Generating GRID Setting $setting with EMA 100k on $generation_shards GPUs"
generation_pids=()
for ((rank=0; rank<generation_shards; rank++)); do
    CUDA_VISIBLE_DEVICES="$rank" "$python_bin" -u -m aligndit.script.eval.infer_grid_semantic_vae \
        --checkpoint "$checkpoint" --step 100000 --output "$output" --shared-manifest "$manifest" \
        --setting "$setting" --rank "$rank" --nshard "$generation_shards" --device cuda:0 \
        > "$output/logs/infer_rank${rank}.log" 2>&1 &
    generation_pids+=("$!")
done
wait_group generation "${generation_pids[@]}"
generated_count="$(find "$output/test" -type f -name '*.wav' | wc -l)"
[[ "$generated_count" -eq 3281 ]] || { echo "Expected 3281 generated WAVs, found $generated_count" >&2; exit 1; }

wavlm="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
asr="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
emotion="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
avhubert_checkpoint="${ROOT_PREFIX}/zjw524/projects/data/large_vox_iter5.pt"
avhubert_fairseq="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq"
avhubert_user_dir="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert"

echo 'Running SPKSIM, EMOSIM, AV-HuBERT extraction and sharded WER'
metric_pids=()
CUDA_VISIBLE_DEVICES=0 "$python_bin" -u -m aligndit.script.eval.eval_grid_generated \
    -e sim -g "$output" --manifest "$manifest" --wavlm-ckpt "$wavlm" \
    > "$output/logs/sim.log" 2>&1 & metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=1 "$python_bin" -u -m aligndit.script.eval.eval_grid_generated \
    -e emosim -g "$output" --manifest "$manifest" --emo-ckpt "$emotion" \
    > "$output/logs/emosim.log" 2>&1 & metric_pids+=("$!")
CUDA_VISIBLE_DEVICES=2 PYTHONPATH="$project_root/src:$avhubert_fairseq" "$python_bin" -u \
    src/aligndit/script/misc/extract_avhubert.py --nshard 1 --rank 0 \
    --v-input-dir "$assets/video/test" --a-input-dir "$output/test" \
    --output-dir "$output/avhubert_feat/test" --ckpt-path "$avhubert_checkpoint" \
    --user_dir "$avhubert_user_dir" > "$output/logs/avhubert.log" 2>&1 & metric_pids+=("$!")
for ((rank=0; rank<wer_shards; rank++)); do
    gpu=$((rank + 3))
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u -m aligndit.script.eval.eval_grid_generated \
        -e wer -g "$output" --manifest "$manifest" --asr-ckpt "$asr" \
        --rank "$rank" --nshard "$wer_shards" > "$output/logs/wer_rank${rank}.log" 2>&1 &
    metric_pids+=("$!")
done
wait_group primary_metrics "${metric_pids[@]}"

CUDA_VISIBLE_DEVICES=2 "$python_bin" -u -m aligndit.script.eval.eval_grid_generated \
    -e avsync -g "$output" --manifest "$manifest" --gt-av-feat "$assets/gt_avhubert_feat" \
    > "$output/logs/avsync.log" 2>&1
"$python_bin" -u -m aligndit.script.eval.finalize_grid_evaluation \
    --output "$output" --manifest "$manifest" \
    --generation-shards "$generation_shards" --wer-shards "$wer_shards" \
    > "$output/logs/finalize.log" 2>&1
cat "$output/metrics_summary.json"
