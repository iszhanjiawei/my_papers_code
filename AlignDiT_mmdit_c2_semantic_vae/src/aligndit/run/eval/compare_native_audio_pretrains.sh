#!/usr/bin/env bash
# Native mel500k vs Semantic-VAE S2c70k, without changing training/model code.
# Run from this experiment's project root. Use setsid for a long/background run.
# Required: RUN_ROOT=/absolute/new/output/directory
# Optional: ROOT_PREFIX=/home GPU_ID=0; argument: prepare | canary | formal | all.
set -euo pipefail

: "${RUN_ROOT:?Set RUN_ROOT to an absolute output directory outside the repository}"
: "${ROOT_PREFIX:=}"
: "${GPU_ID:=0}"
export ROOT_PREFIX
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

ALIGN_PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
SVAE_PYTHON="${ROOT_PREFIX}/zjw524/ENTER/venvs/semantic-vae/bin/python"
EMOTION_MODEL="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
GENERATOR=src/aligndit/script/eval/compare_native_audio_pretrains.py
EVALUATOR=src/aligndit/script/eval/evaluate_native_audio_pretrains.py
RUN_STAGE="${1:-all}"

case "${RUN_STAGE}" in prepare|canary|formal|all) ;; *) echo 'Expected prepare, canary, formal, or all' >&2; exit 2 ;; esac
[[ "${RUN_ROOT}" = /* ]] || { echo 'RUN_ROOT must be absolute' >&2; exit 2; }
[[ -f "${GENERATOR}" && -f "${EVALUATOR}" ]] || { echo 'Run from AlignDiT_mmdit_c2_semantic_vae/' >&2; exit 2; }
[[ -x "${ALIGN_PYTHON}" && -x "${SVAE_PYTHON}" && -d "${EMOTION_MODEL}" ]] || { echo 'Missing runtime or local emotion model' >&2; exit 2; }
mkdir -p "${RUN_ROOT}/logs"

run_logged() {
    local stage_name="$1"
    shift
    local stage_log="${RUN_ROOT}/logs/${stage_name}.log"
    [[ ! -e "${stage_log}" ]] || { echo "Refusing to overwrite log: ${stage_log}" >&2; return 2; }
    "$@" 2>&1 | tee "${stage_log}"
}

prepare() {
    run_logged prepare "${ALIGN_PYTHON}" -u "${GENERATOR}" --output-dir "${RUN_ROOT}" --mode prepare
}

compare() {
    local suffix="$1"
    local -a generation_args=() metric_args=()
    if [[ "${suffix}" = canary2 ]]; then
        generation_args=(--limit 2)
        metric_args=(--canary-limit 2 --output-name metrics_canary2)
    fi
    run_logged "${suffix}_encode_context" "${SVAE_PYTHON}" -u "${GENERATOR}" --output-dir "${RUN_ROOT}" --mode encode-svae-context "${generation_args[@]}"
    run_logged "${suffix}_mel" "${ALIGN_PYTHON}" -u "${GENERATOR}" --output-dir "${RUN_ROOT}" --mode mel "${generation_args[@]}"
    run_logged "${suffix}_svae" "${ALIGN_PYTHON}" -u "${GENERATOR}" --output-dir "${RUN_ROOT}" --mode svae "${generation_args[@]}"
    run_logged "${suffix}_decode" "${SVAE_PYTHON}" -u "${GENERATOR}" --output-dir "${RUN_ROOT}" --mode decode-svae "${generation_args[@]}"
    run_logged "${suffix}_metrics" "${ALIGN_PYTHON}" -u "${EVALUATOR}" --run-root "${RUN_ROOT}" --emotion-model "${EMOTION_MODEL}" "${metric_args[@]}"
}

case "${RUN_STAGE}" in
    prepare) prepare ;;
    canary) compare canary2 ;;
    formal) compare formal ;;
    all) prepare; compare canary2; compare formal ;;
esac
