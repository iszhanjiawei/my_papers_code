#!/usr/bin/env bash
set -euo pipefail

__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do
    __envdir="$(dirname "$__envdir")"
done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"

PROJECT_DIR="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_speaker_embedding"
cd "$PROJECT_DIR"

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
EXP_NAME="finetune_celebvdub_mm_c2_campplus_speaker"
MODEL_NAME="AlignDiT_MMDiT_qknorm_ca_c2_campplus_spk_tail6_finetune"
CKPT_STEP=200000
CKPT_DIR="${ROOT_PREFIX}/zjw524/projects/data/ckpts/${MODEL_NAME}_hifigan_16k_CelebVDub_char"
CKPT_PATH="${CKPT_DIR}/model_${CKPT_STEP}.pt"
WAVLM_CKPT="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
ASR_CKPT="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
EMO_CKPT="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
AVHUBERT_CKPT="${ROOT_PREFIX}/zjw524/projects/data/large_vox_iter5.pt"
FAIRSEQ_ROOT="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq"
AVHUBERT_USER_DIR="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert"
VOCODER_PATH="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000"
GT_AV_FEAT="data/CelebVDub/avhubert_feat"
EXPECTED_SAMPLES=213
GEN_DIR="results/${EXP_NAME}_${CKPT_STEP}/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
RUN_LOG_DIR="logs/c2_campplus_eval_200k"
STEP_LOG="${RUN_LOG_DIR}/checkpoint_${CKPT_STEP}.log"
SUMMARY_PATH="${RUN_LOG_DIR}/metrics_summary.tsv"

mkdir -p "$RUN_LOG_DIR"

for required_path in \
    "$CKPT_PATH" \
    "$WAVLM_CKPT" \
    "$ASR_CKPT" \
    "$EMO_CKPT" \
    "$AVHUBERT_CKPT" \
    "$FAIRSEQ_ROOT" \
    "$AVHUBERT_USER_DIR" \
    "$VOCODER_PATH" \
    "$GT_AV_FEAT"; do
    if [ ! -e "$required_path" ]; then
        echo "Missing required path: $required_path" >&2
        exit 1
    fi
done

visible_gpu_count="$(nvidia-smi -L | wc -l)"
if [ "$visible_gpu_count" -ge 4 ]; then
    GPU_NUMS=4
    CUDA_GPUS="0,1,2,3"
elif [ "$visible_gpu_count" -ge 1 ]; then
    GPU_NUMS=1
    CUDA_GPUS="0"
else
    echo "No CUDA GPU is currently visible." >&2
    exit 1
fi

metric_value() {
    local result_file="$1"
    awk -F': ' 'NF == 2 {value=$2} END {print value}' "$result_file"
}

count_files() {
    local search_dir="$1"
    local filename_pattern="$2"
    if [ ! -d "$search_dir" ]; then
        echo 0
        return
    fi
    find "$search_dir" -type f -name "$filename_pattern" | wc -l
}

metric_complete() {
    local result_file="$1"
    local metric_label="$2"
    [ -f "$result_file" ] && \
        [ "$(grep -c '^{' "$result_file" || true)" -eq "$EXPECTED_SAMPLES" ] && \
        grep -q "^${metric_label}: " "$result_file"
}

echo "[$(date '+%F %T')] Checkpoint ${CKPT_STEP}; using ${GPU_NUMS} GPU(s): ${CUDA_GPUS}." | tee "$STEP_LOG"

wav_count="$(count_files "$GEN_DIR" '*.wav')"
if [ "$wav_count" -ne "$EXPECTED_SAMPLES" ]; then
    echo "[$(date '+%F %T')] Inference checkpoint ${CKPT_STEP}; existing WAVs ${wav_count}/${EXPECTED_SAMPLES}." | tee -a "$STEP_LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_GPUS" \
    OMP_NUM_THREADS=1 \
    NCCL_TIMEOUT=1200 \
    NCCL_IB_DISABLE=1 \
    NCCL_P2P_DISABLE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    "$PYTHON" -u -m accelerate.commands.launch \
        --mixed_precision bf16 \
        --num_processes "$GPU_NUMS" \
        --main_process_port 29611 \
        src/aligndit/script/eval/infer.py \
        -n "$EXP_NAME" \
        -s 0 \
        -t celebvdub_test_s1 \
        -nfe 32 \
        -c "$CKPT_STEP" \
        --cfg_t 5 \
        --cfg_v 2 \
        --ckpt-path "$CKPT_PATH" \
        --vocoder-path "$VOCODER_PATH" 2>&1 | tee -a "$STEP_LOG"
else
    echo "[$(date '+%F %T')] Reusing ${wav_count}/${EXPECTED_SAMPLES} generated WAVs." | tee -a "$STEP_LOG"
fi

if ! metric_complete "${GEN_DIR}/_sim_results.jsonl" SIM; then
    echo "[$(date '+%F %T')] SPKSIM" | tee -a "$STEP_LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_GPUS" PYTHONPATH=src \
    "$PYTHON" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e sim -g "$GEN_DIR" -n "$GPU_NUMS" --wavlm_ckpt "$WAVLM_CKPT" 2>&1 | tee -a "$STEP_LOG"
fi

if ! metric_complete "${GEN_DIR}/_wer_results.jsonl" WER; then
    echo "[$(date '+%F %T')] WER" | tee -a "$STEP_LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_GPUS" PYTHONPATH=src \
    "$PYTHON" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e wer -l en -g "$GEN_DIR" -n "$GPU_NUMS" --asr_ckpt "$ASR_CKPT" 2>&1 | tee -a "$STEP_LOG"
fi

if ! metric_complete "${GEN_DIR}/_emosim_results.jsonl" EMOSIM; then
    echo "[$(date '+%F %T')] EMOSIM" | tee -a "$STEP_LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_GPUS" PYTHONPATH=src \
    "$PYTHON" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e emosim -g "$GEN_DIR" -n "$GPU_NUMS" --emo_ckpt "$EMO_CKPT" 2>&1 | tee -a "$STEP_LOG"
fi

av_feat_count="$(count_files "${GEN_DIR}/avhubert_feat" '*.npy')"
if [ "$av_feat_count" -ne "$EXPECTED_SAMPLES" ]; then
    echo "[$(date '+%F %T')] AV-HuBERT extraction; existing features ${av_feat_count}/${EXPECTED_SAMPLES}." | tee -a "$STEP_LOG"
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0 \
    PYTHONPATH="${FAIRSEQ_ROOT}:src" \
    "$PYTHON" -u src/aligndit/script/misc/extract_avhubert.py \
        --nshard 1 --rank 0 \
        --v-input-dir data/CelebVDub/video_mouth/test/test \
        --a-input-dir "${GEN_DIR}/test" \
        --output-dir "${GEN_DIR}/avhubert_feat/test" \
        --ckpt-path "$AVHUBERT_CKPT" \
        --user_dir "$AVHUBERT_USER_DIR" 2>&1 | tee -a "$STEP_LOG"
fi

if ! metric_complete "${GEN_DIR}/_avsync_results.jsonl" AVSYNC; then
    echo "[$(date '+%F %T')] AVSync" | tee -a "$STEP_LOG"
    CUDA_VISIBLE_DEVICES="$CUDA_GPUS" PYTHONPATH=src \
    "$PYTHON" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e avsync -g "$GEN_DIR" -n "$GPU_NUMS" --gt_av_feat "$GT_AV_FEAT" 2>&1 | tee -a "$STEP_LOG"
fi

sim="$(metric_value "${GEN_DIR}/_sim_results.jsonl")"
wer="$(metric_value "${GEN_DIR}/_wer_results.jsonl")"
emosim="$(metric_value "${GEN_DIR}/_emosim_results.jsonl")"
avsync="$(metric_value "${GEN_DIR}/_avsync_results.jsonl")"
printf 'checkpoint\tSPKSIM\tWER\tEMOSIM\tAVSync\n' > "$SUMMARY_PATH"
printf '%s\t%s\t%s\t%s\t%s\n' "$CKPT_STEP" "$sim" "$wer" "$emosim" "$avsync" | tee -a "$SUMMARY_PATH"
echo "[$(date '+%F %T')] Evaluation complete: ${SUMMARY_PATH}"
