#!/usr/bin/env bash
# Resumable CelebV-Dub Setting-1 inference and four-metric evaluation.
set -euo pipefail

experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$experiment_root"
source "$experiment_root/env.sh"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
checkpoint="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_D1_HunyuanDualCA_AllRoPE_AVHuBERTDualRole_L6_W01_finetune_hifigan_16k_CelebVDub_char/model_150000.pt"
config_name="eval_celebvdub_mm_d1_hunyuan_dual_ca_allrope_avhubert_dual_role"
result_root="results/${config_name}_150000/celebvdub_test_s1/seed0_euler_nfe32_hifigan_16k_ss-1.0_cfgt5.0_cfgv2.0_gt-dur"
test_wavs="$result_root/test"
gpu_list="${EVAL_CUDA_VISIBLE_DEVICES:-0,1,2,3}"
world_size="${EVAL_WORLD_SIZE:-4}"
master_port="${EVAL_MASTER_PORT:-29616}"
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

wav_count="$({ find "$test_wavs" -type f -name '*.wav' 2>/dev/null || true; } | wc -l)"
if [ "$wav_count" -ne 213 ]; then
    echo "Generating 213 WAV files from the 150k EMA checkpoint"
    CUDA_VISIBLE_DEVICES="$gpu_list" "$python_bin" -u -m accelerate.commands.launch \
        --multi_gpu --num_processes "$world_size" --num_machines 1 \
        --mixed_precision bf16 --main_process_port "$master_port" \
        src/aligndit/script/eval/infer.py \
        -n "$config_name" -s 0 -t celebvdub_test_s1 -nfe 32 -o euler -ss -1 \
        -c 150000 --cfg_t 5 --cfg_v 2 --ckpt-path "$checkpoint" \
        --vocoder-path "${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/my_papers_code/hifigan_16k_LRS3/g_01000000"
fi

wav_count="$(find "$test_wavs" -type f -name '*.wav' | wc -l)"
test "$wav_count" -eq 213 || { echo "Expected 213 generated WAVs, found $wav_count" >&2; exit 1; }

if [ ! -s "$result_root/_sim_results.jsonl" ]; then
    echo "Evaluating SPKSIM"
    CUDA_VISIBLE_DEVICES="$gpu_list" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e sim -g "$result_root" -n "$world_size" \
        --wavlm_ckpt "$wavlm_checkpoint" --wavlm_base_ckpt "$wavlm_base_checkpoint"
fi

if [ ! -s "$result_root/_wer_results.jsonl" ]; then
    echo "Evaluating WER"
    CUDA_VISIBLE_DEVICES="$gpu_list" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e wer -l en -g "$result_root" -n "$world_size" --asr_ckpt "$asr_checkpoint"
fi

if [ ! -s "$result_root/_emosim_results.jsonl" ]; then
    echo "Evaluating EMOSIM"
    CUDA_VISIBLE_DEVICES="$gpu_list" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e emosim -g "$result_root" -n "$world_size" --emo_ckpt "$emotion_checkpoint"
fi

feature_count="$({ find "$result_root/avhubert_feat/test" -type f -name '*.npy' 2>/dev/null || true; } | wc -l)"
if [ "$feature_count" -ne 213 ]; then
    echo "Extracting generated audio-video AV-HuBERT features"
    CUDA_VISIBLE_DEVICES="${gpu_list%%,*}" \
    PYTHONPATH="$experiment_root/src:${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq" \
    "$python_bin" -u src/aligndit/script/misc/extract_avhubert.py \
        --nshard 1 --rank 0 \
        --v-input-dir data/CelebVDub/video_mouth/test \
        --a-input-dir "$test_wavs" \
        --output-dir "$result_root/avhubert_feat/test" \
        --ckpt-path "$avhubert_checkpoint" --user_dir "$avhubert_user_dir"
fi

feature_count="$(find "$result_root/avhubert_feat/test" -type f -name '*.npy' | wc -l)"
test "$feature_count" -eq 213 || { echo "Expected 213 AV-HuBERT features, found $feature_count" >&2; exit 1; }

if [ ! -s "$result_root/_avsync_results.jsonl" ]; then
    echo "Evaluating AVSync"
    CUDA_VISIBLE_DEVICES="$gpu_list" "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
        -e avsync -g "$result_root" -n "$world_size" --gt_av_feat data/CelebVDub/avhubert_feat
fi

echo "CelebV-Dub Setting-1 four-metric evaluation completed: $result_root"
