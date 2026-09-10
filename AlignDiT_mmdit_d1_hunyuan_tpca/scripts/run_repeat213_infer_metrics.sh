#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
PROJECT_ROOT="$PWD"
PY="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
DATA="${ROOT_PREFIX}/zjw524/projects/data"
BENCH="${REPEAT213_DATASET:-$DATA/evaluations/celebvdub_train_repeat213_seed666_en_20260910}"
export PYTHONPATH=src CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PATH="$(dirname "$PY"):$PATH"
verify() { "$PY" scripts/validate_tpca_eval.py "$1" "$2" --test-list "$BENCH/clips.lst" --split train --gt-feature-root "$BENCH/CelebVDub/avhubert_feat"; }
metric() {
    local out="$1" task="$2"
    if ! verify "$out" "$task" > /dev/null 2>&1; then
        echo "METRIC $out $task $(date -Is)"
        "$PY" -u src/aligndit/script/eval/eval_celebvdub_test.py -e "$task" -g "$out" -n 1 \
          --test-list "$BENCH/clips.lst" --dataset-root "$BENCH/CelebVDub" --split train \
          --asr_ckpt "$DATA/faster-whisper-large-v3" --emo_ckpt "$DATA/emotion2vec_plus_large" \
          --gt_av_feat "$BENCH/CelebVDub/avhubert_feat"
    fi
    verify "$out" "$task"
}
for spec in d1_tpca_150000 d1_tpca_200000 c2_svae_speaker_200000; do
    out="$BENCH/results/$spec"
    step="${spec##*_}"
    mkdir -p "$out"
    if ! verify "$out" wav > /dev/null 2>&1; then
        echo "INFER $spec $(date -Is)"
        if [[ "$spec" == d1_tpca_* ]]; then
            checkpoint="$DATA/ckpts/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_TPCA_6MM12A_CTC6_12_finetune_hifigan_16k_CelebVDub_char/model_${step}.pt"
            "$PY" -u src/aligndit/script/eval/infer.py -n finetune_celebvdub_mm_d1_hunyuan_tpca -c "$step" -s 0 -t celebvdub_test_s1 \
                --ckpt-path "$checkpoint" --vocoder-path "$PROJECT_ROOT/../hifigan_16k_LRS3/g_01000000" \
                --test-list "$BENCH/clips.lst" --dataset-root "$BENCH/CelebVDub" --split train --output-dir "$out"
        else
            checkpoint="$DATA/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_40hz_CelebVDub_char/model_${step}.pt"
            (
                cd "$PROJECT_ROOT/../AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding"
                PYTHONPATH=src "$PY" -u src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py \
                    --checkpoint "$checkpoint" --step "$step" --output-dir "$out" --seed 0 --nfe 32 --cfg-text 5 --cfg-video 2 \
                    --test-list "$BENCH/clips.lst" --manifest "$BENCH/manifest.jsonl" --split train
            )
        fi
    fi
    verify "$out" wav
    for task in sim wer emosim; do metric "$out" "$task"; done
    echo "AUDIO_METRICS_COMPLETE $spec $(date -Is)"
done
mkdir -p "$BENCH/results/ground_truth"
if ! verify "$BENCH/results/ground_truth" wer >/dev/null 2>&1; then
    "$PY" -u src/aligndit/script/eval/eval_celebvdub_test.py -e wer -g "$BENCH/results/ground_truth" -n 1 \
        --test-list "$BENCH/clips.lst" --dataset-root "$BENCH/CelebVDub" --split train --eval_ground_truth --asr_ckpt "$DATA/faster-whisper-large-v3"
fi
verify "$BENCH/results/ground_truth" wer
echo "INFERENCE_AND_AUDIO_METRICS_COMPLETE $(date -Is)"
