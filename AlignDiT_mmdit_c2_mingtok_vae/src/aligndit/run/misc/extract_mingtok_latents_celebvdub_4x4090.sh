#!/usr/bin/env bash
set -euo pipefail

# Extract the 79,613 CelebV-Dub train clips with one frozen MingTok encoder per
# GPU.  Re-running this command is safe: valid FP32 arrays are reused and each
# new array is installed atomically.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
if [[ -f "${PROJECT_DIR}/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "${PROJECT_DIR}/env.sh"
fi

ROOT_PREFIX="${ROOT_PREFIX:-}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python}"
AUDIO_ROOT="${AUDIO_ROOT:-${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/audio}"
METADATA_PATH="${METADATA_PATH:-${ROOT_PREFIX}/zjw524/projects/data/CelebVDub_char/raw.arrow}"
CACHE_DIR="${CACHE_DIR:-${ROOT_PREFIX}/zjw524/projects/data/CelebVDub_mingtok_acoustic_64d_sample_seed666_fp32}"
MINGTOK_REPO="${MINGTOK_REPO:-${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/MingTok-VAE/paper_code/MingTok-Audio}"
MINGTOK_CHECKPOINT="${MINGTOK_CHECKPOINT:-${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/MingTok-VAE/checkpoint/MingTok-Audio}"
SPLIT="${SPLIT:-train}"
EXPECTED_COUNT="${EXPECTED_COUNT:-79613}"
BASE_SEED="${BASE_SEED:-666}"
BACKEND="${BACKEND:-eager}"
DTYPE="${DTYPE:-bfloat16}"
GPU_IDS="${GPU_IDS:-0 1 2 3}"
read -r -a GPUS <<< "${GPU_IDS//,/ }"
NSHARD="${#GPUS[@]}"

if [[ "${NSHARD}" -ne 4 ]]; then
    echo "Expected four GPU IDs for the 4x4090 extraction, got: ${GPU_IDS}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 2
fi

EXTRACTOR="${PROJECT_DIR}/src/aligndit/script/misc/extract_mingtok_latents.py"
mkdir -p "${CACHE_DIR}/logs"

COMMON_ARGS=(
    --audio-root "${AUDIO_ROOT}"
    --selection metadata
    --metadata-path "${METADATA_PATH}"
    --cache-dir "${CACHE_DIR}"
    --split "${SPLIT}"
    --extension .wav
    --expected-count "${EXPECTED_COUNT}"
    --repo-path "${MINGTOK_REPO}"
    --checkpoint-dir "${MINGTOK_CHECKPOINT}"
    --nshard "${NSHARD}"
    --posterior-mode sample
    --base-seed "${BASE_SEED}"
    --backend "${BACKEND}"
    --dtype "${DTYPE}"
)

pids=()
cleanup() {
    trap - INT TERM EXIT
    for pid in "${pids[@]:-}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "${pid}" 2>/dev/null || true
        fi
    done
}
trap cleanup INT TERM EXIT

echo "MingTok CelebV-Dub extraction"
echo "  audio_root=${AUDIO_ROOT}"
echo "  metadata_path=${METADATA_PATH} (exact C2 train selection)"
echo "  cache_dir=${CACHE_DIR}"
echo "  GPUs=${GPUS[*]}"
echo "  expected_count=${EXPECTED_COUNT}"
echo "  posterior=sample, raw FP32 [T,64], seed=${BASE_SEED}"

# Perform the expensive 79,613-path existence check exactly once. Rank workers
# and the merge pass reconstruct the same metadata order without repeating all
# filesystem stats; each rank's torchaudio.load still checks its own files.
echo "Running one metadata/count/path preflight before GPU launch."
(
    cd "${PROJECT_DIR}"
    PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -u "${EXTRACTOR}" \
        "${COMMON_ARGS[@]}" \
        --validate-selection-only
) 2>&1 | tee "${CACHE_DIR}/logs/metadata_preflight_${SPLIT}.log"

for rank in "${!GPUS[@]}"; do
    gpu="${GPUS[$rank]}"
    log="${CACHE_DIR}/logs/extract_${SPLIT}_rank${rank}.log"
    echo "Launching rank ${rank}/${NSHARD} on physical GPU ${gpu}; log=${log}"
    (
        cd "${PROJECT_DIR}"
        CUDA_VISIBLE_DEVICES="${gpu}" \
        OMP_NUM_THREADS=1 \
        PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        "${PYTHON_BIN}" -u "${EXTRACTOR}" \
            "${COMMON_ARGS[@]}" \
            --rank "${rank}" \
            --skip-source-file-check \
            --device cuda
    ) >"${log}" 2>&1 &
    pids+=("$!")
done

failed=0
for rank in "${!pids[@]}"; do
    if ! wait "${pids[$rank]}"; then
        failed=1
        echo "Rank ${rank} failed; tail of log:" >&2
        tail -n 80 "${CACHE_DIR}/logs/extract_${SPLIT}_rank${rank}.log" >&2 || true
    fi
done
if [[ "${failed}" -ne 0 ]]; then
    echo "At least one shard failed. Re-run this script; completed .npy files will be reused." >&2
    exit 1
fi

trap - INT TERM EXIT
echo "All four shards completed; validating and merging manifests."
(
    cd "${PROJECT_DIR}"
    PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -u "${EXTRACTOR}" \
        "${COMMON_ARGS[@]}" \
        --skip-source-file-check \
        --merge-only
) 2>&1 | tee "${CACHE_DIR}/logs/merge_${SPLIT}.log"

echo "MingTok cache complete:"
echo "  ${CACHE_DIR}/latents/${SPLIT}/<speaker>/<clip>.npy"
echo "  ${CACHE_DIR}/manifest.jsonl"
echo "  ${CACHE_DIR}/contract.json"
