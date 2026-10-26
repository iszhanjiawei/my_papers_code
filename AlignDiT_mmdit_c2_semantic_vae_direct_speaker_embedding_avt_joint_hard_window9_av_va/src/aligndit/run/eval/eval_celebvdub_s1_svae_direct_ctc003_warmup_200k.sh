#!/bin/bash
# Inference + evaluation: Direct-C2 CTC=0.03 warmup, step 200k, CelebV-Dub Setting 1.

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
if [[ ! -f "$project_root/env.sh" ]]; then
    echo "Cannot locate project env.sh from $script_dir" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_c2_ctc003_warmup10k30k_40hz_CelebVDub_char"
checkpoint="$checkpoint_dir/model_200000.pt"
config="src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_ctc003_warmup.yaml"
step=200000
output_dir="$checkpoint_dir/eval_s1_200000"
eval_gpu="${EVAL_GPU:-0}"

celebvdub="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/alignDiT_baseline/AlignDiT/data/CelebVDub"
gt_av_feat="$celebvdub/avhubert_feat"
test_list="${ROOT_PREFIX}/zjw524/projects/data/celebvdub_test_s1.lst"
wavlm="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
asr="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
emo="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
avhubert_ckpt="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt"
avhubert_user_dir="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert"
avhubert_fairseq="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq"

if [[ ! -f "$checkpoint" || -L "$checkpoint" ]]; then
    echo "Missing regular checkpoint: $checkpoint" >&2
    exit 1
fi
if [[ -d "$output_dir" ]] && find "$output_dir" -type f | grep -q .; then
    echo "Refusing to overwrite non-empty evaluation directory: $output_dir" >&2
    exit 1
fi

echo "===== [1/5] Generate 213 Semantic-VAE waveforms on GPU ${eval_gpu} ====="
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py \
    --checkpoint "$checkpoint" \
    --step "$step" \
    --config "$config" \
    --output-dir "$output_dir" \
    --test-list "$test_list" \
    --semantic-vae-repo "${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE" \
    --semantic-vae-checkpoint "${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k" \
    --device cuda:0

echo "===== [2/5] SPKSIM ====="
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e sim -g "$output_dir" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --wavlm_ckpt "$wavlm"

echo "===== [3/5] WER ====="
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e wer -l en -g "$output_dir" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --asr_ckpt "$asr"

echo "===== [4/5] EMOSIM ====="
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e emosim -g "$output_dir" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --emo_ckpt "$emo"

echo "===== [5/5] AVSync feature extraction ====="
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH="$avhubert_fairseq:src" \
"$python_bin" -u src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 --rank 0 \
    --v-input-dir "$celebvdub/video_mouth/test/test" \
    --a-input-dir "$output_dir/test" \
    --output-dir "$output_dir/avhubert_feat/test" \
    --ckpt-path "$avhubert_ckpt" \
    --user_dir "$avhubert_user_dir"

echo "===== [5/5] AVSync metric ====="
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e avsync -g "$output_dir" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --gt_av_feat "$gt_av_feat"

echo "===== ALL DONE: Direct-C2 CTC=0.03 warmup @ step ${step} ====="
