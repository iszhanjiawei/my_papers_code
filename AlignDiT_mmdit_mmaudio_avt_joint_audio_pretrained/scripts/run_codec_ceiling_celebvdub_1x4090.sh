#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"
source env.sh

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
DATA_ROOT="${ROOT_PREFIX}/zjw524/projects/data"
OUTPUT_ROOT="${1:-${DATA_ROOT}/codec_ceiling_celebvdub}"
FAIRSEQ_ROOT="${DATA_ROOT}/av_hubert/fairseq/fairseq"
AVHUBERT_USER_DIR="${DATA_ROOT}/av_hubert/avhubert/avhubert"
AVHUBERT_CKPT="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt"
VIDEO_ROOT="${DATA_ROOT}/CelebVDub/video_mouth/test"
GT_AV_FEAT="${DATA_ROOT}/CelebVDub/gt_eval/avhubert_feat"

CODEC_DIRS=(
  mel_hifigan
  acoustic_vae_dim64_sample
  semantic_vae_600k_sample
  semantic_vae_1000k_sample
)

env PYTHONPATH=src "${PYTHON}" -u \
  src/aligndit/script/eval/reconstruct_celebvdub_codecs.py \
  --latent-mode sample \
  --output-root "${OUTPUT_ROOT}"

for codec_dir in "${CODEC_DIRS[@]}"; do
  gen_dir="${OUTPUT_ROOT}/${codec_dir}"

  if [[ ! -f "${gen_dir}/_sim_results.jsonl" ]]; then
    env PYTHONPATH=src "${PYTHON}" -u \
      src/aligndit/script/eval/eval_celebvdub_test.py \
      --eval_task sim \
      --gen_wav_dir "${gen_dir}" \
      --gpu_nums 1 \
      --wavlm_ckpt "${DATA_ROOT}/wavlm_large_finetune.pth" \
      --wavlm_base_ckpt "${DATA_ROOT}/wavlm_large_s3prl.pt"
  fi

  if [[ ! -f "${gen_dir}/_wer_results.jsonl" ]]; then
    env PYTHONPATH=src "${PYTHON}" -u \
      src/aligndit/script/eval/eval_celebvdub_test.py \
      --eval_task wer \
      --gen_wav_dir "${gen_dir}" \
      --gpu_nums 1 \
      --asr_ckpt "${DATA_ROOT}/faster-whisper-large-v3"
  fi

  if [[ ! -f "${gen_dir}/_emosim_results.jsonl" ]]; then
    env PYTHONPATH=src "${PYTHON}" -u \
      src/aligndit/script/eval/eval_celebvdub_test.py \
      --eval_task emosim \
      --gen_wav_dir "${gen_dir}" \
      --gpu_nums 1 \
      --emo_ckpt "${DATA_ROOT}/emotion2vec_plus_large"
  fi

  env PYTHONPATH="${FAIRSEQ_ROOT}:src" "${PYTHON}" -u \
    src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 \
    --rank 0 \
    --v-input-dir "${VIDEO_ROOT}" \
    --a-input-dir "${gen_dir}" \
    --output-dir "${gen_dir}/avhubert_feat" \
    --ckpt-path "${AVHUBERT_CKPT}" \
    --user_dir "${AVHUBERT_USER_DIR}"

  if [[ ! -f "${gen_dir}/_avsync_results.jsonl" ]]; then
    env PYTHONPATH=src "${PYTHON}" -u \
      src/aligndit/script/eval/eval_celebvdub_test.py \
      --eval_task avsync \
      --gen_wav_dir "${gen_dir}" \
      --gpu_nums 1 \
      --gt_av_feat "${GT_AV_FEAT}"
  fi
done

env PYTHONPATH=src "${PYTHON}" -u \
  src/aligndit/script/eval/summarize_codec_ceiling.py \
  "${OUTPUT_ROOT}"
