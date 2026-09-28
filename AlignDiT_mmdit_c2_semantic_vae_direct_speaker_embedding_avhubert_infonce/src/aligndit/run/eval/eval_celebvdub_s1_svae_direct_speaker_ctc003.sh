#!/usr/bin/env bash
# Generate S1 speech and evaluate SPKSIM, WER, EMOSIM and AVSync.
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
if [[ ! -f "$project_root/env.sh" ]]; then
    echo "Cannot locate this experiment's env.sh" >&2
    exit 1
fi
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_40hz_CelebVDub_char"
export CKPT_STEP="${CKPT_STEP:-200000}"
export CFG_VIDEO="${CFG_VIDEO:-2.0}"
export EVAL_GPU="${EVAL_GPU:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-$checkpoint_dir/eval_s1_${CKPT_STEP}_speaker_cfgv${CFG_VIDEO}}"
export OMP_NUM_THREADS=1

celebvdub="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/alignDiT_baseline/AlignDiT/data/CelebVDub"
test_list="${ROOT_PREFIX}/zjw524/projects/data/celebvdub_test_s1.lst"
wavlm="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
asr="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
emo="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
avhubert_ckpt="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt"
avhubert_user_dir="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert"
avhubert_fairseq="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq"

bash "$script_dir/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$project_root/src" \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e sim -g "$OUTPUT_DIR" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --wavlm_ckpt "$wavlm"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$project_root/src" \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e wer -l en -g "$OUTPUT_DIR" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --asr_ckpt "$asr"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$project_root/src" \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e emosim -g "$OUTPUT_DIR" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --emo_ckpt "$emo"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$project_root/src:$avhubert_fairseq" \
"$python_bin" -u src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 --rank 0 \
    --v-input-dir "$celebvdub/video_mouth/test/test" \
    --a-input-dir "$OUTPUT_DIR/test" \
    --output-dir "$OUTPUT_DIR/avhubert_feat/test" \
    --ckpt-path "$avhubert_ckpt" \
    --user_dir "$avhubert_user_dir"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" PYTHONPATH="$project_root/src" \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e avsync -g "$OUTPUT_DIR" -n 1 \
    --test-list "$test_list" --celebvdub-root "$celebvdub" --gt_av_feat "$celebvdub/avhubert_feat"
echo "Evaluation complete: $OUTPUT_DIR"
