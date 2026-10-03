#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../../.." && pwd)"
# shellcheck source=/dev/null
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
manifest_dir="$project_root/data_chem/svae1000k_sample_seed666_fp32/manifests"
audio_root="${ROOT_PREFIX}/zjw524/projects/aligndit_project_gird/aligndit_project_chem/alignDiT_baseline/AlignDiT/data_chem_v2/Chem/audio"
campplus_checkpoint="$project_root/data_chem/pretrained_models/campplus/campplus_cn_en_common.pt"
wavlm_model_dir="${ROOT_PREFIX}/zjw524/projects/data/wavlm-base-plus"

: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to an allocated GPU before feature extraction}"
export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"

inventory_sha="$(sha256sum "$manifest_dir/inventory.jsonl" | cut -d ' ' -f 1)"
train_sha="$(sha256sum "$manifest_dir/train.jsonl" | cut -d ' ' -f 1)"

"$python_bin" -u src/aligndit/script/misc/extract_campplus_speaker_embeddings.py \
    --manifest "$manifest_dir/inventory.jsonl" \
    --expected-manifest-sha256 "$inventory_sha" --expected-count 6328 \
    --audio-root "$audio_root" \
    --cache-dir "$project_root/data_chem/campplus_spk_emb_zh_en_16k" \
    --checkpoint "$campplus_checkpoint" \
    --batch-utterances "${CAMPPLUS_BATCH_SIZE:-4}" --num-workers "${TEACHER_NUM_WORKERS:-2}"

exec "$python_bin" -u src/aligndit/script/misc/extract_wavlm_base_plus_repa.py \
    --manifest "$manifest_dir/train.jsonl" \
    --expected-manifest-sha256 "$train_sha" --expected-count 5821 \
    --audio-root "$audio_root" \
    --cache-dir "$project_root/data_chem/wavlm_base_plus_repa_final_fp16" \
    --local-model-dir "$wavlm_model_dir" \
    --max-batch-items "${WAVLM_BATCH_SIZE:-2}" --max-batch-seconds "${WAVLM_BATCH_SECONDS:-20}" \
    --max-duration-seconds 30
