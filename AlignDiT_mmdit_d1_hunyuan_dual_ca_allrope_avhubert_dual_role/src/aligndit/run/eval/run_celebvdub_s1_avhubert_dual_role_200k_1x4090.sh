#!/usr/bin/env bash
# Resumable single-GPU CelebV-Dub Setting-1 evaluation for the 200k EMA checkpoint.
set -euo pipefail

experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$experiment_root"
source "$experiment_root/env.sh"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
checkpoint="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_AVHuBERTDualRole_L6_W01_finetune_hifigan_16k_CelebVDub_char/model_200000.pt"
config_name="eval_celebvdub_mm_d1_hunyuan_dual_ca_allrope_avhubert_dual_role"
result_root="results/${config_name}_200000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1.0_cfgt5.0_cfgv2.0_gt-dur"
test_wavs="$result_root/test"
gpu_id="${EVAL_GPU:-0}"
expected_items=213
wavlm_checkpoint="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
wavlm_base_checkpoint="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_s3prl.pt"
asr_checkpoint="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
emotion_checkpoint="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
avhubert_checkpoint="${ROOT_PREFIX}/zjw524/projects/data/large_vox_iter5.pt"
avhubert_user_dir="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert"

for required in "$checkpoint" "$wavlm_checkpoint" "$wavlm_base_checkpoint" "$avhubert_checkpoint"; do
    test -f "$required" || { echo "Missing required file: $required" >&2; exit 1; }
done
for required_dir in "$asr_checkpoint" "$emotion_checkpoint" data/CelebVDub; do
    test -d "$required_dir" || { echo "Missing required directory: $required_dir" >&2; exit 1; }
done

export PYTHONPATH="$experiment_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1

count_files() {
    local directory="$1"
    local pattern="$2"
    if [ ! -d "$directory" ]; then
        echo 0
        return
    fi
    find "$directory" -type f -name "$pattern" | wc -l
}

metric_complete() {
    local result_file="$1"
    local label="$2"
    [ -f "$result_file" ] && tail -n 1 "$result_file" | grep -q "^${label}: "
}

wav_count="$(count_files "$test_wavs" '*.wav')"
if [ "$wav_count" -ne "$expected_items" ]; then
    echo "Generating ${expected_items} WAV files from the 200k EMA checkpoint on GPU ${gpu_id}"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u src/aligndit/script/eval/infer.py \
        -n "$config_name" -s 0 -t celebvdub_test_s1 -nfe 32 -o euler -ss -1 \
        -c 200000 --cfg_t 5 --cfg_v 2 --ckpt-path "$checkpoint" \
        --vocoder-path "${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000"
fi

wav_count="$(count_files "$test_wavs" '*.wav')"
test "$wav_count" -eq "$expected_items" || { echo "Expected ${expected_items} generated WAVs, found $wav_count" >&2; exit 1; }

if ! metric_complete "$result_root/_sim_results.jsonl" SIM; then
    echo "Evaluating SPKSIM"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e sim -g "$result_root" -n 1 --wavlm_ckpt "$wavlm_checkpoint" --wavlm_base_ckpt "$wavlm_base_checkpoint"
fi
if ! metric_complete "$result_root/_wer_results.jsonl" WER; then
    echo "Evaluating WER"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e wer -l en -g "$result_root" -n 1 --asr_ckpt "$asr_checkpoint"
fi
if ! metric_complete "$result_root/_emosim_results.jsonl" EMOSIM; then
    echo "Evaluating EMOSIM"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e emosim -g "$result_root" -n 1 --emo_ckpt "$emotion_checkpoint"
fi

feature_count="$(count_files "$result_root/avhubert_feat" '*.npy')"
if [ "$feature_count" -ne "$expected_items" ]; then
    echo "Extracting generated audio-video AV-HuBERT features"
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    PYTHONPATH="$experiment_root/src:${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq" \
    "$python_bin" -u src/aligndit/script/misc/extract_avhubert.py \
        --nshard 1 --rank 0 \
        --v-input-dir data/CelebVDub/video_mouth/test/test \
        --a-input-dir "$test_wavs" \
        --output-dir "$result_root/avhubert_feat/test" \
        --ckpt-path "$avhubert_checkpoint" --user_dir "$avhubert_user_dir"
fi

feature_count="$(count_files "$result_root/avhubert_feat" '*.npy')"
test "$feature_count" -eq "$expected_items" || { echo "Expected ${expected_items} AV-HuBERT features, found $feature_count" >&2; exit 1; }

if ! metric_complete "$result_root/_avsync_results.jsonl" AVSYNC; then
    echo "Evaluating AVSync"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e avsync -g "$result_root" -n 1 --gt_av_feat data/CelebVDub/avhubert_feat
fi

echo "CelebV-Dub Setting-1 four-metric evaluation completed: $result_root"
