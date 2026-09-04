#!/bin/bash
set -euo pipefail

# Reproducible and resumable single-GPU CelebV-Dub Setting-1 evaluation for
# the fresh D1 ctc_lambda=0.3 checkpoints at update 150k and update 185k.

eval_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$eval_root"

# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$eval_root"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------

GPU_ID="${EVAL_GPU:-0}"
PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
EXP_NAME=finetune_celebvdub_mm_d1_6mm12audio_dual_ctc6_12_ctc03_fresh
CKPT_DIR="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_d1_6mm_12audio_dual_ctc6_12_ctc03_fresh_finetune_hifigan_16k_CelebVDub_char"
EXPECTED_ITEMS=213
WAVLM_CKPT="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
ASR_CKPT="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
EMO_CKPT="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
AVHUBERT_CKPT="${ROOT_PREFIX}/zjw524/projects/data/large_vox_iter5.pt"
AVHUBERT_USER_DIR="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert"
AVHUBERT_FAIRSEQ="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq"
VOCODER_PATH="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000"
GT_AV_FEAT=data/CelebVDub/avhubert_feat

count_files() {
    local dir="$1"
    local pattern="$2"
    if [ ! -d "$dir" ]; then
        echo 0
        return
    fi
    find "$dir" -type f -name "$pattern" | wc -l
}

metric_complete() {
    local result_file="$1"
    local label="$2"
    [ -f "$result_file" ] && tail -n 1 "$result_file" | grep -q "^${label}: "
}

run_metric() {
    local metric="$1"
    local result_dir="$2"
    local result_file="$result_dir/_${metric}_results.jsonl"
    local label
    local extra_args=()

    case "$metric" in
        sim)
            label=SIM
            extra_args=(--wavlm_ckpt "$WAVLM_CKPT")
            ;;
        wer)
            label=WER
            extra_args=(-l en --asr_ckpt "$ASR_CKPT")
            ;;
        emosim)
            label=EMOSIM
            extra_args=(--emo_ckpt "$EMO_CKPT")
            ;;
        avsync)
            label=AVSYNC
            extra_args=(--gt_av_feat "$GT_AV_FEAT")
            ;;
        *)
            echo "Unknown metric: $metric" >&2
            exit 1
            ;;
    esac

    if metric_complete "$result_file" "$label"; then
        echo "SKIP $metric: complete result exists ($(tail -n 1 "$result_file"))"
        return
    fi

    echo "===== ${label} ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONPATH=src \
        "$PYTHON_BIN" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e "$metric" -g "$result_dir" -n 1 "${extra_args[@]}"
}

evaluate_checkpoint() {
    local result_step="$1"
    local checkpoint_name="$2"
    local checkpoint_path="$CKPT_DIR/$checkpoint_name"
    local result_dir="results/${EXP_NAME}_${result_step}/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
    local wav_count
    local feat_count

    echo "===== CHECKPOINT ${checkpoint_name} (update ${result_step}) ====="
    if [ ! -f "$checkpoint_path" ]; then
        echo "Missing checkpoint: $checkpoint_path" >&2
        exit 1
    fi

    wav_count="$(count_files "$result_dir/test" '*.wav')"
    if [ "$wav_count" -eq "$EXPECTED_ITEMS" ]; then
        echo "SKIP inference: found $wav_count/$EXPECTED_ITEMS generated wav files"
    else
        CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src \
            "$PYTHON_BIN" -u src/aligndit/script/eval/infer.py \
            -n "$EXP_NAME" \
            -s 0 \
            -t celebvdub_test_s1 \
            -nfe 32 \
            -c "$result_step" \
            --cfg_t 5 \
            --cfg_v 2 \
            --vocoder-path "$VOCODER_PATH" \
            --ckpt-path "$checkpoint_path"
    fi

    wav_count="$(count_files "$result_dir/test" '*.wav')"
    if [ "$wav_count" -ne "$EXPECTED_ITEMS" ]; then
        echo "Inference completeness failure: $wav_count/$EXPECTED_ITEMS wav files" >&2
        exit 1
    fi

    run_metric sim "$result_dir"
    run_metric wer "$result_dir"
    run_metric emosim "$result_dir"

    feat_count="$(count_files "$result_dir/avhubert_feat" '*.npy')"
    if [ "$feat_count" -eq "$EXPECTED_ITEMS" ]; then
        echo "SKIP AV-HuBERT extraction: found $feat_count/$EXPECTED_ITEMS features"
    else
        echo "===== AVSync: extract generated-audio AV-HuBERT features ====="
        CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 \
        PYTHONPATH="src:${AVHUBERT_FAIRSEQ}" \
            "$PYTHON_BIN" -u src/aligndit/script/misc/extract_avhubert.py \
            --nshard 1 --rank 0 \
            --v-input-dir data/CelebVDub/video_mouth/test/test \
            --a-input-dir "$result_dir/test" \
            --output-dir "$result_dir/avhubert_feat/test" \
            --ckpt-path "$AVHUBERT_CKPT" \
            --user_dir "$AVHUBERT_USER_DIR"
    fi

    feat_count="$(count_files "$result_dir/avhubert_feat" '*.npy')"
    if [ "$feat_count" -ne "$EXPECTED_ITEMS" ]; then
        echo "AV-HuBERT completeness failure: $feat_count/$EXPECTED_ITEMS features" >&2
        exit 1
    fi

    run_metric avsync "$result_dir"
    echo "CelebV-Dub Setting-1 evaluation completed for ${checkpoint_name} (update ${result_step})."
}

case "${1:-all}" in
    150000)
        evaluate_checkpoint 150000 model_150000.pt
        ;;
    185000|last)
        evaluate_checkpoint 185000 model_last.pt
        ;;
    all)
        evaluate_checkpoint 150000 model_150000.pt
        evaluate_checkpoint 185000 model_last.pt
        ;;
    *)
        echo "Usage: $0 [150000|185000|last|all]" >&2
        exit 2
        ;;
esac
