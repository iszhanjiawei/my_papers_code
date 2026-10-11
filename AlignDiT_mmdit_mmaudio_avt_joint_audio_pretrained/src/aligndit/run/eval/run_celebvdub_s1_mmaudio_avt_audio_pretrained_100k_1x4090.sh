#!/usr/bin/env bash
set -euo pipefail

# Resumable CelebV-Dub Setting-1 evaluation for a selected checkpoint of the
# MMAudio-style AVT Joint D1 model initialized from the audio-only pretrained
# model. This uses the same seed/sampler/CFG settings as the baseline runs.
# EVAL_STEP defaults to 100000 and can be overridden, for example 150000.

eval_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$eval_root"

if [ -f env.sh ]; then
    # shellcheck disable=SC1091
    source env.sh
fi

GPU_ID="${EVAL_GPU:-0}"
PYTHON_BIN="${ROOT_PREFIX:-}/zjw524/ENTER/envs/aligndit/bin/python"
EXPECTED_ITEMS=213
EXP_NAME=finetune_celebvdub_mmaudio_avt_joint_d1_audio_pretrained
STEP="${EVAL_STEP:-100000}"
if ! [[ "$STEP" =~ ^[0-9]+$ ]]; then
    echo "EVAL_STEP must be a positive integer, got: $STEP" >&2
    exit 1
fi
CKPT="${ROOT_PREFIX:-}/zjw524/projects/data/ckpts/AlignDiT_MMAudioMMDiT_AVTJoint_D1_6J12A_DualCTC6_12_AudioPretrained_LibriSpeech500k_hifigan_16k_CelebVDub_char/model_${STEP}.pt"
VOCODER="${ROOT_PREFIX:-}/zjw524/projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000"
WAVLM_CKPT="${ROOT_PREFIX:-}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
ASR_CKPT="${ROOT_PREFIX:-}/zjw524/projects/data/faster-whisper-large-v3"
EMO_CKPT="${ROOT_PREFIX:-}/zjw524/projects/data/emotion2vec_plus_large"
AVHUBERT_CKPT="${ROOT_PREFIX:-}/zjw524/projects/data/large_vox_iter5.pt"
AVHUBERT_USER_DIR="${ROOT_PREFIX:-}/zjw524/projects/data/av_hubert/avhubert/avhubert"
AVHUBERT_FAIRSEQ="${ROOT_PREFIX:-}/zjw524/projects/data/av_hubert/fairseq/fairseq"
GT_AV_FEAT=data/CelebVDub/avhubert_feat
RESULT_DIR="results/${EXP_NAME}_${STEP}/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"

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
    local label="$2"
    shift 2
    local result_file="${RESULT_DIR}/_${metric}_results.jsonl"

    if metric_complete "$result_file" "$label"; then
        echo "SKIP ${label}: $(tail -n 1 "$result_file")"
        return
    fi

    echo "===== ${label} ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONPATH=src \
        "$PYTHON_BIN" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e "$metric" -g "$RESULT_DIR" -n 1 "$@"
}

for required in "$PYTHON_BIN" "$CKPT" "$VOCODER" data/celebvdub_test_s1.lst; do
    if [ ! -e "$required" ]; then
        echo "Required evaluation input is missing: $required" >&2
        exit 1
    fi
done

wav_count="$(count_files "$RESULT_DIR/test" '*.wav')"
if [ "$wav_count" -eq "$EXPECTED_ITEMS" ]; then
    echo "SKIP inference: found ${wav_count}/${EXPECTED_ITEMS} generated wav files"
else
    echo "===== INFERENCE: checkpoint ${STEP} ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 PYTHONPATH=src \
        "$PYTHON_BIN" -u src/aligndit/script/eval/infer.py \
        -n "$EXP_NAME" -s 0 -t celebvdub_test_s1 -nfe 32 -c "$STEP" \
        --cfg_t 5 --cfg_v 2 --ckpt-path "$CKPT" --vocoder-path "$VOCODER"
fi

wav_count="$(count_files "$RESULT_DIR/test" '*.wav')"
if [ "$wav_count" -ne "$EXPECTED_ITEMS" ]; then
    echo "Inference completeness failure: ${wav_count}/${EXPECTED_ITEMS} wav files" >&2
    exit 1
fi

run_metric sim SIM --wavlm_ckpt "$WAVLM_CKPT"
run_metric wer WER -l en --asr_ckpt "$ASR_CKPT"
run_metric emosim EMOSIM --emo_ckpt "$EMO_CKPT"

feat_count="$(count_files "$RESULT_DIR/avhubert_feat" '*.npy')"
if [ "$feat_count" -eq "$EXPECTED_ITEMS" ]; then
    echo "SKIP AV-HuBERT extraction: found ${feat_count}/${EXPECTED_ITEMS} features"
else
    echo "===== AVSync: extract generated-audio AV-HuBERT features ====="
    CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=1 \
    PYTHONPATH="src:${AVHUBERT_FAIRSEQ}" \
        "$PYTHON_BIN" -u src/aligndit/script/misc/extract_avhubert.py \
        --nshard 1 --rank 0 \
        --v-input-dir data/CelebVDub/video_mouth/test/test \
        --a-input-dir "$RESULT_DIR/test" \
        --output-dir "$RESULT_DIR/avhubert_feat/test" \
        --ckpt-path "$AVHUBERT_CKPT" \
        --user_dir "$AVHUBERT_USER_DIR"
fi

feat_count="$(count_files "$RESULT_DIR/avhubert_feat" '*.npy')"
if [ "$feat_count" -ne "$EXPECTED_ITEMS" ]; then
    echo "AV-HuBERT completeness failure: ${feat_count}/${EXPECTED_ITEMS} features" >&2
    exit 1
fi

run_metric avsync AVSYNC --gt_av_feat "$GT_AV_FEAT"

echo "CelebV-Dub Setting-1 evaluation completed for audio-pretrained checkpoint ${STEP}."
