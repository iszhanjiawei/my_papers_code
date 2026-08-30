#!/usr/bin/env bash
# Run CelebV-Dub Setting-1 EMA inference and evaluation for the stable MingTok C2 checkpoints.
# The pipeline is resumable: a stage is skipped only after all 213 expected per-item artifacts exist.

set -euo pipefail

# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------

PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
DATA_DIR="${ROOT_PREFIX}/zjw524/projects/data"
DATASET_ROOT="${DATA_DIR}/CelebVDub"
TEST_LIST="${DATA_DIR}/celebvdub_test_s1.lst"
CKPT_DIR="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_mingtok_64d50hz_scratch_ctc001_warmup10k30k_seed666_mingtok_64d50hz_CelebVDub_char"
CONFIG_PATH="src/aligndit/config/train_celebvdub_mingtok_c2.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-100000 150000}"
EXPECTED_ITEMS="$(grep -cve '^[[:space:]]*$' "$TEST_LIST")"

WAVLM_CKPT="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
WAVLM_BASE_CKPT="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_s3prl.pt"
ASR_CKPT="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
EMO_CKPT="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
AVHUBERT_CKPT="${ROOT_PREFIX}/zjw524/projects/data/large_vox_iter5.pt"
AVHUBERT_USER_DIR="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert"
AVHUBERT_FAIRSEQ="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq"
GT_AV_FEAT="${DATASET_ROOT}/avhubert_feat"

for required in \
    "$PYTHON_BIN" "$TEST_LIST" "$DATASET_ROOT" "$CONFIG_PATH" \
    "$WAVLM_CKPT" "$WAVLM_BASE_CKPT" "$ASR_CKPT" "$EMO_CKPT" \
    "$AVHUBERT_CKPT" "$AVHUBERT_USER_DIR" "$AVHUBERT_FAIRSEQ" "$GT_AV_FEAT"; do
    if [ ! -e "$required" ]; then
        echo "Required evaluation asset not found: $required" >&2
        exit 2
    fi
done

mkdir -p logs "$OUTPUT_ROOT"

count_files() {
    local root="$1"
    local pattern="$2"
    if [ ! -d "$root" ]; then
        echo 0
        return 0
    fi
    find "$root" -type f -name "$pattern" 2>/dev/null | wc -l
}

metric_complete() {
    local result_file="$1"
    local label="$2"
    [ -f "$result_file" ] \
        && [ "$(grep -c '^{' "$result_file")" -eq "$EXPECTED_ITEMS" ] \
        && grep -q "^${label}:" "$result_file"
}

run_logged() {
    local stage="$1"
    local log_path="$2"
    shift 2
    echo "[$(date '+%F %T %Z')] START $stage"
    "$@" 2>&1 | tee "$log_path"
    echo "[$(date '+%F %T %Z')] DONE  $stage"
}

echo "GPU_ID=$GPU_ID"
echo "STEPS=$STEPS"
echo "EXPECTED_ITEMS=$EXPECTED_ITEMS"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true

for step in $STEPS; do
    checkpoint="${CKPT_DIR}/model_${step}.pt"
    if [ ! -f "$checkpoint" ]; then
        echo "Checkpoint not found: $checkpoint" >&2
        exit 2
    fi

    run_name="train_celebvdub_mingtok_c2_ctc001_warmup10k30k_seed666_model_${step}"
    result_dir="${OUTPUT_ROOT}/${run_name}/celebvdub_test_s1/seed0_euler_nfe32_epss_sway1_cfgt5_cfgv2_gt-dur"
    mkdir -p "$result_dir"

    wav_count="$(count_files "$result_dir" '*.wav')"
    if [ "$wav_count" -eq "$EXPECTED_ITEMS" ]; then
        echo "SKIP inference model_${step}: found $wav_count/$EXPECTED_ITEMS waveforms"
    else
        run_logged "inference model_${step}" "logs/infer_celebvdub_mingtok_ctc001_model_${step}.log" \
            env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src \
            "$PYTHON_BIN" -u src/aligndit/script/eval/infer_celebvdub_mingtok_s1.py \
            --checkpoint "$checkpoint" \
            --config "$CONFIG_PATH" \
            --output-dir "$result_dir" \
            --seed 0 \
            --posterior-seed 666 \
            --nfe 32 \
            --ode-method euler \
            --sway 1.0 \
            --cfg-text 5.0 \
            --cfg-video 2.0 \
            --codec-dtype bfloat16 \
            --codec-backend eager
    fi

    wav_count="$(count_files "$result_dir" '*.wav')"
    if [ "$wav_count" -ne "$EXPECTED_ITEMS" ]; then
        echo "Inference completeness failure for model_${step}: $wav_count/$EXPECTED_ITEMS waveforms" >&2
        exit 1
    fi

    for task in sim wer emosim; do
        label="${task^^}"
        result_file="${result_dir}/_${task}_results.jsonl"
        if metric_complete "$result_file" "$label"; then
            echo "SKIP $task model_${step}: complete result exists"
            continue
        fi

        extra_args=()
        case "$task" in
            sim)
                extra_args=(--wavlm_ckpt "$WAVLM_CKPT" --wavlm_base_ckpt "$WAVLM_BASE_CKPT")
                ;;
            wer)
                extra_args=(-l en --asr_ckpt "$ASR_CKPT")
                ;;
            emosim)
                extra_args=(--emo_ckpt "$EMO_CKPT")
                ;;
        esac
        run_logged "$task model_${step}" "logs/eval_celebvdub_mingtok_ctc001_model_${step}_${task}.log" \
            env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src \
            "$PYTHON_BIN" -u src/aligndit/script/eval/eval_celebvdub_test.py \
            -e "$task" -g "$result_dir" -n 1 \
            --test-list "$TEST_LIST" --dataset-root "$DATASET_ROOT" \
            "${extra_args[@]}"
    done

    generated_av_feat="${result_dir}/avhubert_feat"
    av_feat_count="$(count_files "$generated_av_feat" '*.npy')"
    if [ "$av_feat_count" -eq "$EXPECTED_ITEMS" ]; then
        echo "SKIP AV-HuBERT extraction model_${step}: found $av_feat_count/$EXPECTED_ITEMS features"
    else
        run_logged "AV-HuBERT extraction model_${step}" \
            "logs/eval_celebvdub_mingtok_ctc001_model_${step}_avhubert_extract.log" \
            env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
            PYTHONPATH="src:${AVHUBERT_FAIRSEQ}" \
            "$PYTHON_BIN" -u src/aligndit/script/misc/extract_avhubert.py \
            --nshard 1 --rank 0 \
            --v-input-dir "${DATASET_ROOT}/video_mouth/test" \
            --a-input-dir "${result_dir}/test" \
            --output-dir "${generated_av_feat}/test" \
            --ckpt-path "$AVHUBERT_CKPT" \
            --user_dir "$AVHUBERT_USER_DIR"
    fi

    av_feat_count="$(count_files "$generated_av_feat" '*.npy')"
    if [ "$av_feat_count" -ne "$EXPECTED_ITEMS" ]; then
        echo "AV-HuBERT completeness failure for model_${step}: $av_feat_count/$EXPECTED_ITEMS features" >&2
        exit 1
    fi

    avsync_file="${result_dir}/_avsync_results.jsonl"
    if metric_complete "$avsync_file" "AVSYNC"; then
        echo "SKIP avsync model_${step}: complete result exists"
    else
        run_logged "avsync model_${step}" "logs/eval_celebvdub_mingtok_ctc001_model_${step}_avsync.log" \
            env CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src \
            "$PYTHON_BIN" -u src/aligndit/script/eval/eval_celebvdub_test.py \
            -e avsync -g "$result_dir" -n 1 \
            --test-list "$TEST_LIST" --dataset-root "$DATASET_ROOT" \
            --gt_av_feat "$GT_AV_FEAT"
    fi

    summary_path="${result_dir}/_summary.txt"
    {
        echo "checkpoint=$checkpoint"
        echo "samples=$EXPECTED_ITEMS"
        echo "sampling=EMA seed0 euler nfe32 epss sway1 cfg_text5 cfg_video2 gt_duration"
        for task in sim wer emosim avsync; do
            grep -E '^[A-Z]+:' "${result_dir}/_${task}_results.jsonl"
        done
    } > "$summary_path"
    echo "Summary model_${step}: $summary_path"
    cat "$summary_path"
done

echo "[$(date '+%F %T %Z')] ALL CHECKPOINT EVALUATIONS COMPLETE"
