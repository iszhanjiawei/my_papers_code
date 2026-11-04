#!/bin/bash
# Evaluate one cfg_video value for Semantic-VAE Direct-C2 CTC=0.03 @ 200k.

set -euo pipefail

cfg_video="${1:?usage: $0 CFG_VIDEO}"
cfg_text="${CFG_TEXT:-5.0}"
eval_gpu="${EVAL_GPU:-0}"

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
cfg_tag="${cfg_video//./p}"
if [[ "$cfg_video" == "2" || "$cfg_video" == "2.0" ]]; then
    output_dir="$checkpoint_dir/eval_s1_200000"
else
    output_dir="$checkpoint_dir/eval_s1_200000_cfgt5_cfgv${cfg_tag}"
fi

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

if [[ ! -f "$output_dir/inference_summary.json" ]]; then
    if [[ -d "$output_dir" ]] && find "$output_dir" -type f -print -quit | grep -q .; then
        echo "Refusing to mix with incomplete non-empty directory: $output_dir" >&2
        exit 1
    fi
    echo "===== Generate: cfg_text=$cfg_text cfg_video=$cfg_video ====="
    CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
    "$python_bin" -u src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py \
        --checkpoint "$checkpoint" --step 200000 --config "$config" \
        --output-dir "$output_dir" --test-list "$test_list" \
        --semantic-vae-repo "${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE" \
        --semantic-vae-checkpoint "${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k" \
        --device cuda:0 --seed 0 --nfe 32 --sway -1.0 \
        --cfg-text "$cfg_text" --cfg-video "$cfg_video"
else
    echo "===== Reuse completed inference: $output_dir ====="
fi

run_metric() {
    local task="$1"
    local result="$output_dir/_${task}_results.jsonl"
    shift
    if [[ -f "$result" ]] && tail -n 1 "$result" | grep -q "^[A-Z].*:"; then
        echo "===== Reuse metric: $task ====="
        return
    fi
    echo "===== Evaluate: $task ====="
    CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
    "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e "$task" -g "$output_dir" -n 1 \
        --test-list "$test_list" --celebvdub-root "$celebvdub" "$@"
}

run_metric sim --wavlm_ckpt "$wavlm"
run_metric wer -l en --asr_ckpt "$asr"
run_metric emosim --emo_ckpt "$emo"
run_metric emoembed --emo_ckpt "$emo"

if [[ ! -f "$output_dir/_avsync_results.jsonl" ]] || ! tail -n 1 "$output_dir/_avsync_results.jsonl" | grep -q '^AVSYNC:'; then
    echo "===== Extract AV-HuBERT features ====="
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH="$avhubert_fairseq:src" \
    "$python_bin" -u src/aligndit/script/misc/extract_avhubert.py \
        --nshard 1 --rank 0 \
        --v-input-dir "$celebvdub/video_mouth/test/test" \
        --a-input-dir "$output_dir/test" \
        --output-dir "$output_dir/avhubert_feat/test" \
        --ckpt-path "$avhubert_ckpt" --user_dir "$avhubert_user_dir"
    run_metric avsync --gt_av_feat "$gt_av_feat"
else
    echo "===== Reuse metric: avsync ====="
fi

echo "===== DONE: cfg_text=$cfg_text cfg_video=$cfg_video output=$output_dir ====="
