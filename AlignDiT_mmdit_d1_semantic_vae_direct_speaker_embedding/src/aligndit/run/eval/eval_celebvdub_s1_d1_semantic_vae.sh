#!/usr/bin/env bash
# D1 Semantic-VAE: inference and the historical WER / SIM / EMOSIM / AVSync metrics.
# Usage: bash this_script.sh [150000|last|/path/to/model.pt] [new_output_directory]
# Long runs: setsid env PYTHONUNBUFFERED=1 bash this_script.sh ... > logs/eval.log 2>&1 &
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../../.." && pwd)"
# shellcheck source=/dev/null
source "$project_root/env.sh"
cd "$project_root"
export PYTHONUNBUFFERED=1

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
config="${CONFIG:-src/aligndit/config/finetune_celebvdub_mm_d1_semantic_vae_direct.yaml}"
checkpoint_dir="${CHECKPOINT_DIR:-${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_d1_semantic_vae_direct_ctc003_warmup10k30k_40hz_CelebVDub_char}"
selector="${1:-last}"
step_args=()
if [[ "$selector" =~ ^[0-9]+$ ]]; then
    checkpoint="$checkpoint_dir/model_${selector}.pt"
    step_args=(--step "$selector")
elif [[ "$selector" == last ]]; then
    checkpoint="$checkpoint_dir/model_last.pt"
else
    checkpoint="$selector"
fi
output_dir="${2:-$checkpoint_dir/eval_s1_$(basename "${checkpoint%.pt}")_seed0_nfe32_cfgt5_cfgv2}"
eval_gpu="${EVAL_GPU:-0}"
celebvdub="${CELEBVDUB_ROOT:-${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/alignDiT_baseline/AlignDiT/data/CelebVDub}"
test_list="${TEST_LIST:-${ROOT_PREFIX}/zjw524/projects/data/celebvdub_test_s1.lst}"
wavlm="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_finetune.pth"
wavlm_base="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/wavlm_large_s3prl.pt"
asr="${ROOT_PREFIX}/zjw524/projects/data/faster-whisper-large-v3"
emo="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
avhubert_ckpt="${ROOT_PREFIX}/zjw524/alignDiT_pretrain_models/large_vox_iter5.pt"
avhubert_user_dir="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/avhubert/avhubert"
avhubert_fairseq="${ROOT_PREFIX}/zjw524/projects/data/av_hubert/fairseq/fairseq"

for resource in "$python_bin" "$config" "$checkpoint" "$test_list" "$wavlm" "$wavlm_base" \
    "$asr" "$emo" "$avhubert_ckpt" "$avhubert_user_dir" "$avhubert_fairseq"; do
    if [[ ! -e "$resource" ]]; then
        echo "Missing evaluation resource: $resource" >&2
        exit 1
    fi
done

# Reject incomplete test inputs before starting the expensive inference stages.
"$python_bin" - "$test_list" "$celebvdub" "$output_dir" <<'PY'
import sys
from pathlib import Path

test_list, root, output = map(Path, sys.argv[1:])
clips = [line.strip() for line in test_list.read_text().splitlines() if line.strip()]
if len(clips) != 213 or len(set(clips)) != 213:
    raise RuntimeError("Expected 213 unique CelebV-Dub Setting 1 test clips")
if output.exists() and any(output.iterdir()):
    raise FileExistsError(f"Choose a new output directory; this one is not empty: {output}")
for clip in clips:
    for folder, suffix in (("audio/test", ".wav"), ("text/test", ".txt"),
                           ("video_mouth/test/test", ".mp4"), ("avhubert_feat/test", ".npy")):
        path = root / folder / (clip + suffix)
        if not path.is_file():
            raise FileNotFoundError(path)
print("Preflight passed: 213/213 reference inputs; empty output directory")
PY

echo "[1/6] D1 Semantic-VAE inference: $checkpoint -> $output_dir"
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py \
    --checkpoint "$checkpoint" "${step_args[@]}" --config "$config" \
    --output-dir "$output_dir" --test-list "$test_list" --seed 0 --nfe 32 \
    --cfg-text 5 --cfg-video 2 --sway -1 --device cuda:0

echo "[2/6] SIM"
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e sim -g "$output_dir" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --wavlm_ckpt "$wavlm" --wavlm_base_ckpt "$wavlm_base"

echo "[3/6] WER"
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e wer -l en -g "$output_dir" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --asr_ckpt "$asr"

echo "[4/6] EMOSIM"
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e emosim -g "$output_dir" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --emo_ckpt "$emo"

echo "[5/6] AV-HuBERT features"
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH="src:$avhubert_fairseq" \
"$python_bin" -u src/aligndit/script/misc/extract_avhubert.py \
    --nshard 1 --rank 0 --v-input-dir "$celebvdub/video_mouth/test/test" \
    --a-input-dir "$output_dir/test" --output-dir "$output_dir/avhubert_feat/test" \
    --ckpt-path "$avhubert_ckpt" --user_dir "$avhubert_user_dir"

echo "[6/6] AVSync"
CUDA_VISIBLE_DEVICES="$eval_gpu" PYTHONPATH=src \
"$python_bin" -u src/aligndit/script/eval/eval_celebvdub_test.py \
    -e avsync -g "$output_dir" -n 1 --test-list "$test_list" --celebvdub-root "$celebvdub" \
    --gt_av_feat "$celebvdub/avhubert_feat"

"$python_bin" - "$test_list" "$output_dir" <<'PY'
import json
import math
import sys
from pathlib import Path

test_list, output = map(Path, sys.argv[1:])
clips = [line.strip() for line in test_list.read_text().splitlines() if line.strip()]
summary = json.loads((output / "inference_summary.json").read_text())
if summary["dataset"]["count"] != 213 or len(summary["outputs"]) != 213:
    raise RuntimeError("Incomplete inference summary")
for clip in clips:
    for path in (output / "test" / (clip + ".wav"), output / "avhubert_feat/test" / (clip + ".npy")):
        if not path.is_file():
            raise FileNotFoundError(path)
for metric in ("sim", "wer", "emosim", "avsync"):
    lines = [line for line in (output / f"_{metric}_results.jsonl").read_text().splitlines() if line.strip()]
    if len(lines) != 214 or not lines[-1].startswith(metric.upper() + ": "):
        raise RuntimeError(f"Incomplete {metric} results")
    for line in lines[:-1]:
        json.loads(line)
    if not math.isfinite(float(lines[-1].split(": ", 1)[1])):
        raise RuntimeError(f"Non-finite {metric} result")
    print(lines[-1])
print(f"Complete: 213/213 waveforms, features and metric rows; EMA update {summary['checkpoint']['update']}")
PY
echo "Evaluation complete: $output_dir"
