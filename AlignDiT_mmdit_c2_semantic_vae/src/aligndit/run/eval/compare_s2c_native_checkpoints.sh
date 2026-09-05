#!/usr/bin/env bash
# Run in AlignDiT_mmdit_c2_semantic_vae; background runs must use setsid.
# The two candidate workers use distinct GPUs; no training is launched.
set -euo pipefail
: "${ROOT_PREFIX:=}"
: "${STUDY_ROOT:?Set a NEW absolute output directory outside the repository}"
: "${REFERENCE_ROOT:?Set the completed native mel500k/S2c70k v2 output directory}"
: "${GPU_50:=2}"
: "${GPU_60:=3}"
[[ "${GPU_50}" != "${GPU_60}" ]] || { echo 'Use distinct GPUs for the candidate workers' >&2; exit 2; }
export ROOT_PREFIX OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1 PYTHONPATH=src
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
ALIGN_PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
SVAE_PYTHON="${ROOT_PREFIX}/zjw524/ENTER/venvs/semantic-vae/bin/python"
EMOTION_MODEL="${ROOT_PREFIX}/zjw524/projects/data/emotion2vec_plus_large"
GEN=src/aligndit/script/eval/compare_native_audio_pretrains.py
EVAL=src/aligndit/script/eval/evaluate_native_audio_pretrains.py
STUDY=src/aligndit/script/eval/compare_s2c_native_checkpoints.py

"${ALIGN_PYTHON}" -u "${STUDY}" --mode prepare --reference-root "${REFERENCE_ROOT}" --study-root "${STUDY_ROOT}"
mkdir "${STUDY_ROOT}/logs"
CUDA_VISIBLE_DEVICES="${GPU_50}" "${ALIGN_PYTHON}" -u "${GEN}" --mode svae --svae-update 70000 --limit 2 --output-dir "${STUDY_ROOT}/replay_70k" > "${STUDY_ROOT}/logs/replay70k.log" 2>&1
"${ALIGN_PYTHON}" -u "${STUDY}" --mode check-replay --reference-root "${REFERENCE_ROOT}" --study-root "${STUDY_ROOT}"

candidate() {
    local update="$1" gpu="$2"
    local root="${STUDY_ROOT}/s2c_$((update / 1000))k"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    # Check each actual intermediate checkpoint end to end, without scoring
    # the canary or changing/re-encoding the shared posterior context.
    "${ALIGN_PYTHON}" -u "${GEN}" --mode svae --svae-update "${update}" --limit 2 --output-dir "${root}"
    "${SVAE_PYTHON}" -u "${GEN}" --mode decode-svae --svae-update "${update}" --limit 2 --output-dir "${root}"
    "${ALIGN_PYTHON}" -u "${EVAL}" --svae-update "${update}" --run-root "${root}" --canary-limit 2 --output-name metrics_canary2 --validate-only
    "${ALIGN_PYTHON}" -u "${GEN}" --mode svae --svae-update "${update}" --output-dir "${root}"
    "${SVAE_PYTHON}" -u "${GEN}" --mode decode-svae --svae-update "${update}" --output-dir "${root}"
    "${ALIGN_PYTHON}" -u "${EVAL}" --svae-update "${update}" --run-root "${root}" --emotion-model "${EMOTION_MODEL}"
}

candidate 50000 "${GPU_50}" > "${STUDY_ROOT}/logs/s2c_50k.log" 2>&1 &
pid50=$!
candidate 60000 "${GPU_60}" > "${STUDY_ROOT}/logs/s2c_60k.log" 2>&1 &
pid60=$!
printf 'candidate_50k_pid=%s candidate_60k_pid=%s\n' "${pid50}" "${pid60}"
status50=0 status60=0
wait "${pid50}" || status50=$?
wait "${pid60}" || status60=$?
[[ "${status50}" = 0 && "${status60}" = 0 ]] || { echo "Candidate failed: 50k=${status50}, 60k=${status60}" >&2; exit 1; }
"${ALIGN_PYTHON}" -u "${STUDY}" --mode summarize --reference-root "${REFERENCE_ROOT}" --study-root "${STUDY_ROOT}"
