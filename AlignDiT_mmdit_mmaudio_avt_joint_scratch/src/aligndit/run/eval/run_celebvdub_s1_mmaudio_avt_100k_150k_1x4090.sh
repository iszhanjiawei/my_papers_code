#!/bin/bash
set -euo pipefail

# Reproducible and resumable single-GPU CelebV-Dub Setting-1 evaluation for
# the scratch MMAudio-style AVT Joint DiT checkpoints at 100k, 150k, and 200k.

eval_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$eval_root"

GPU_ID="${EVAL_GPU:-0}"
PYTHON_BIN=/zjw524/ENTER/envs/aligndit/bin/python
EXPECTED_ITEMS=213
WAVLM_CKPT=/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth
ASR_CKPT=/zjw524/projects/data/faster-whisper-large-v3
EMO_CKPT=/zjw524/projects/data/emotion2vec_plus_large
AVHUBERT_CKPT=/zjw524/projects/data/large_vox_iter5.pt
AVHUBERT_USER_DIR=/zjw524/projects/data/av_hubert/avhubert/avhubert
AVHUBERT_FAIRSEQ=/zjw524/projects/data/av_hubert/fairseq/fairseq
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
    local step="$1"
    local infer_script="$2"
    local result_dir="results/finetune_celebvdub_mmaudio_avt_joint_d1_scratch_${step}/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
    local wav_count
    local feat_count

    echo "===== CHECKPOINT ${step} ====="
    wav_count="$(count_files "$result_dir/test" '*.wav')"
    if [ "$wav_count" -eq "$EXPECTED_ITEMS" ]; then
        echo "SKIP inference: found $wav_count/$EXPECTED_ITEMS generated wav files"
    else
        bash "$infer_script"
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
}

evaluate_checkpoint 100000 src/aligndit/run/eval/infer_celebvdub_s1_idea6_mmdit_100k.sh
evaluate_checkpoint 150000 src/aligndit/run/eval/infer_celebvdub_s1_idea6_mmdit_150k.sh
evaluate_checkpoint 200000 src/aligndit/run/eval/infer_celebvdub_s1_idea6_mmdit_full_200k.sh

echo "CelebV-Dub Setting-1 evaluation completed for checkpoints 100000, 150000, and 200000."
