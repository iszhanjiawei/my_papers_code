#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
source env.sh
STEP="${1:?Usage: bash run_celebvdub_s1_tpca.sh 150000|200000}"
case "$STEP" in 150000|200000) ;; *) exit 2;; esac
PY="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
DATA="${ROOT_PREFIX}/zjw524/projects/data"
EXP=finetune_celebvdub_mm_d1_hunyuan_tpca
CKPT="$DATA/ckpts/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_TPCA_6MM12A_CTC6_12_finetune_hifigan_16k_CelebVDub_char/model_${STEP}.pt"
OUT="results/${EXP}_${STEP}/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1_cfgt5.0_cfgv2.0_gt-dur"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 PYTHONPATH=src
export CUDA_VISIBLE_DEVICES="${INFER_GPUS:-0,1}"
METRIC_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
check() { "$PY" scripts/validate_tpca_eval.py "$OUT" "$1"; }
echo "Starting TPCA update=$STEP GPUs=$CUDA_VISIBLE_DEVICES at $(date -Is)"
if ! check wav; then
  "$PY" -u -m accelerate.commands.launch --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no --main_process_port "${EVAL_PORT:-29581}" \
    src/aligndit/script/eval/infer.py -n "$EXP" -c "$STEP" -s 0 -t celebvdub_test_s1 -nfe 32 --cfg_t 5 --cfg_v 2 \
    --ckpt-path "$CKPT" --vocoder-path "${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000"
fi
check wav
export CUDA_VISIBLE_DEVICES="$METRIC_GPU"
for metric in sim wer emosim; do
  if ! check "$metric"; then
    echo "Starting $metric at $(date -Is)"
    "$PY" -u src/aligndit/script/eval/eval_celebvdub_test.py -e "$metric" -g "$OUT" -n 1 \
      --asr_ckpt "$DATA/faster-whisper-large-v3" --emo_ckpt "$DATA/emotion2vec_plus_large"
  fi
  check "$metric"
done
if ! check features; then
  echo "Starting AV-HuBERT extraction at $(date -Is)"
  PYTHONPATH="src:$DATA/av_hubert/fairseq/fairseq" "$PY" -u src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 --rank 0 --v-input-dir "$DATA/CelebVDub/video_mouth/test/test" --a-input-dir "$OUT/test" \
    --output-dir "$OUT/avhubert_feat/test" --ckpt-path "$DATA/large_vox_iter5.pt" --user_dir "$DATA/av_hubert/avhubert/avhubert"
fi
check features
if ! check avsync; then
  "$PY" -u src/aligndit/script/eval/eval_celebvdub_test.py -e avsync -g "$OUT" -n 1 --gt_av_feat "$DATA/CelebVDub/avhubert_feat"
fi
check all
echo "COMPLETE update=$STEP at $(date -Is)"
