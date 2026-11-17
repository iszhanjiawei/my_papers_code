#!/usr/bin/env bash
# Same historical S1 protocol, sharing GPUs with the continuing four-GPU run.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
checkpoint_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char"
output="${OUTPUT_DIR:-${ROOT_PREFIX}/zjw524/projects/data/evaluations/CelebVDub_avt_fixed13_libritts_repa_s1_150k_seed0_nfe32_cfg5_2}"
celebvdub="${ROOT_PREFIX}/zjw524/projects/data/CelebVDub"
test_list="${ROOT_PREFIX}/zjw524/projects/data/celebvdub_test_s1.lst"
export PYTHONPATH="$project_root/src"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$output/evaluation_logs"
if [[ -e "$output/inference_summary.json" ]] || [[ -d "$output/test" ]]; then
    echo "Refusing to overwrite an existing inference: $output" >&2
    exit 1
fi
stage_log="$output/evaluation_logs"
printf 'Start S1 150k EMA: %s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES=3 "$python_bin" -u scripts/run_setting2.py --memory-fraction 0.16 \
    --tool src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py \
    --checkpoint "$checkpoint_dir/model_150000.pt" --step 150000 \
    --config src/aligndit/config/finetune_celebvdub_repa.yaml --output-dir "$output" \
    --cfg-text 5.0 --cfg-video 2.0 --seed 0 --nfe 32 --sway -1 \
    > "$stage_log/inference.log" 2>&1
printf 'Inference complete: %s\n' "$(date --iso-8601=seconds)"
CUDA_VISIBLE_DEVICES=1 "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e sim -g "$output" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --wavlm_ckpt "${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth" \
    > "$stage_log/sim.log" 2>&1 &
sim_pid=$!
CUDA_VISIBLE_DEVICES=2 "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e emosim -g "$output" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --emo_ckpt "${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large" \
    > "$stage_log/emosim.log" 2>&1 &
emo_pid=$!
CUDA_VISIBLE_DEVICES=3 "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e wer -l en -g "$output" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --asr_ckpt "${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3" \
    > "$stage_log/wer.log" 2>&1 &
wer_pid=$!
# Reuse GPU1 only after SPKSIM releases it; WER/EMOSIM keep independent GPUs.
wait "$sim_pid"
avhubert_fairseq="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq"
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$project_root/src:$avhubert_fairseq" \
    "$python_bin" -u scripts/run_setting2.py --memory-fraction 0.16 \
    --tool src/aligndit/script/misc/extract_avhubert.py --nshard 1 --rank 0 \
    --v-input-dir "$celebvdub/video_mouth/test/test" --a-input-dir "$output/test" \
    --output-dir "$output/avhubert_feat/test" \
    --ckpt-path "${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt" \
    --user_dir "${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert" \
    > "$stage_log/extract_avsync.log" 2>&1
CUDA_VISIBLE_DEVICES=1 "$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e avsync -g "$output" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --gt_av_feat "$celebvdub/avhubert_feat" > "$stage_log/avsync.log" 2>&1
wait "$emo_pid"
wait "$wer_pid"
CUDA_VISIBLE_DEVICES='' "$python_bin" -u scripts/audit_avt_joint_s1.py \
    --output-dir "$output" --test-list "$test_list" --gt-root "$celebvdub" \
    > "$stage_log/audit.log" 2>&1
printf 'All stages and 213-clip audit complete: %s %s\n' "$(date --iso-8601=seconds)" "$output"
