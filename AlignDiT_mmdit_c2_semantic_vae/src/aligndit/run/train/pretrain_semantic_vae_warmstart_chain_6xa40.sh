#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

if [[ -z "${ROOT_PREFIX}" && ! -x "/zjw524/ENTER/envs/aligndit/bin/python" ]]; then
    for candidate_prefix in /home /s7home; do
        if [[ -x "${candidate_prefix}/zjw524/ENTER/envs/aligndit/bin/python" ]]; then
            export ROOT_PREFIX="${candidate_prefix}"
            break
        fi
    done
fi

PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
STAGE_LAUNCHER="${PROJECT_ROOT}/src/aligndit/run/train/pretrain_semantic_vae_warmstart_6xa40.sh"
VALIDATOR="${PROJECT_ROOT}/src/aligndit/script/misc/validate_semantic_vae_warmstart_checkpoint.py"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "AlignDiT Python executable does not exist: ${PYTHON_BIN}" >&2
    exit 2
fi

START_STAGE=${1:-${WARMSTART_CHAIN_START_STAGE:-s1}}
case "${START_STAGE}" in
    s1) STAGES=(s1 s2a s2b s2c) ;;
    s2a) STAGES=(s2a s2b s2c) ;;
    s2b) STAGES=(s2b s2c) ;;
    s2c) STAGES=(s2c) ;;
    *)
        echo "Start stage must be one of s1, s2a, s2b, s2c; got: ${START_STAGE}" >&2
        exit 2
        ;;
esac

GPU_IDS=${GPU_IDS:-2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
FRAME_BUDGET_PER_GPU=${FRAME_BUDGET_PER_GPU:-7200}
MAX_SAMPLES=${MAX_SAMPLES:-32}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29630}
METRIC_INTERVAL_SECONDS=${METRIC_INTERVAL_SECONDS:-10}
CHAIN_RUN_ID=${WARMSTART_CHAIN_RUN_ID:-$(date +%Y%m%dT%H%M%S)}
if [[ ! "${CHAIN_RUN_ID}" =~ ^[A-Za-z0-9_.-]+$ ]]; then
    echo "WARMSTART_CHAIN_RUN_ID contains unsafe characters: ${CHAIN_RUN_ID}" >&2
    exit 2
fi
for value_name in NUM_PROCESSES FRAME_BUDGET_PER_GPU MAX_SAMPLES MAIN_PROCESS_PORT METRIC_INTERVAL_SECONDS; do
    value=${!value_name}
    if [[ ! "${value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "${value_name} must be a positive integer, got: ${value}" >&2
        exit 2
    fi
done
IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if (( ${#GPU_ARRAY[@]} != NUM_PROCESSES )); then
    echo "GPU_IDS must contain exactly NUM_PROCESSES=${NUM_PROCESSES} entries, got: ${GPU_IDS}" >&2
    exit 2
fi

CHECKPOINT_ROOT="${ROOT_PREFIX}/zjw524/projects/data/ckpts"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMING_LOG="${LOG_DIR}/warmstart_chain_timing_${CHAIN_RUN_ID}.csv"
METRICS_LOG="${LOG_DIR}/warmstart_chain_gpu_metrics_${CHAIN_RUN_ID}.csv"
mkdir -p "${CHECKPOINT_ROOT}" "${LOG_DIR}"

LOCK_PATH="${CHECKPOINT_ROOT}/.aligndit_semantic_vae_warmstart_chain_6xa40.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "Another Semantic-VAE warm-start chain holds ${LOCK_PATH}" >&2
    exit 3
fi

if [[ ! -s "${TIMING_LOG}" ]]; then
    printf 'run_id,stage,event,timestamp,epoch_seconds,elapsed_seconds,target_update,exit_code\n' > "${TIMING_LOG}"
fi
if [[ ! -s "${METRICS_LOG}" ]]; then
    printf 'timestamp,stage,gpu_index,memory_used_mib,memory_total_mib,utilization_gpu_percent\n' > "${METRICS_LOG}"
fi

MONITOR_PID=""
STAGE_PID=""
stop_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
    MONITOR_PID=""
}

stop_stage() {
    if [[ -n "${STAGE_PID}" ]] && kill -0 "${STAGE_PID}" 2>/dev/null; then
        kill -- "-${STAGE_PID}" 2>/dev/null || true
        wait "${STAGE_PID}" 2>/dev/null || true
    fi
    STAGE_PID=""
}

cleanup() {
    stop_monitor
    stop_stage
}

handle_signal() {
    local exit_code=$1
    cleanup
    exit "${exit_code}"
}

trap cleanup EXIT
trap 'handle_signal 129' HUP
trap 'handle_signal 130' INT
trap 'handle_signal 143' TERM

record_event() {
    local stage=$1
    local event=$2
    local timestamp=$3
    local epoch_seconds=$4
    local elapsed_seconds=$5
    local target_update=$6
    local exit_code=$7
    printf '%s,%s,%s,%s,%s,%s,%s,%s\n' \
        "${CHAIN_RUN_ID}" "${stage}" "${event}" "${timestamp}" "${epoch_seconds}" \
        "${elapsed_seconds}" "${target_update}" "${exit_code}" >> "${TIMING_LOG}"
}

assert_target_gpus_free() {
    local gpu_id
    local active_pids
    for gpu_id in "${GPU_ARRAY[@]}"; do
        if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
            echo "GPU_IDS must contain non-negative integer indices, got: ${GPU_IDS}" >&2
            exit 2
        fi
        if ! nvidia-smi -i "${gpu_id}" --query-gpu=index --format=csv,noheader,nounits >/dev/null 2>&1; then
            echo "GPU index does not exist or is inaccessible: ${gpu_id}" >&2
            exit 2
        fi
        active_pids=$(nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
            | sed '/^[[:space:]]*$/d' || true)
        if [[ -n "${active_pids}" ]]; then
            echo "Refusing to start because GPU ${gpu_id} is occupied by PID(s): ${active_pids//$'\n'/,}" >&2
            exit 4
        fi
    done
}

active_warmstart_processes() {
    ps -eo pid=,comm=,args= | awk '
        $2 ~ /^(python|python3|python3[.]10|pt_main_thread)$/ &&
        $0 ~ /pretrain_semantic_vae_warmstart[.]py/ {print}
    '
}

start_monitor() {
    local stage=$1
    local chain_pid=$$
    (
        exec 9>&-
        while kill -0 "${chain_pid}" 2>/dev/null; do
            timestamp=$(date '+%F %T')
            nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                --format=csv,noheader,nounits \
                | awk -v ts="${timestamp}" -v st="${stage}" -F', ' \
                    '{printf "%s,%s,%s,%s,%s,%s\n",ts,st,$1,$2,$3,$4}'
            sleep "${METRIC_INTERVAL_SECONDS}"
        done
    ) >> "${METRICS_LOG}" 2>&1 &
    MONITOR_PID=$!
}

stage_horizon() {
    if [[ $1 == s2c ]]; then
        printf '70000\n'
    else
        printf '10000\n'
    fi
}

stage_checkpoint_dir() {
    printf '%s/AlignDiT_SemanticVAE_mel_warmstart_%s_40hz_LibriSpeech\n' "${CHECKPOINT_ROOT}" "$1"
}

validate_stage() {
    local stage=$1
    local target_update=$2
    local horizon=$3
    local checkpoint_dir
    checkpoint_dir=$(stage_checkpoint_dir "${stage}")
    local checkpoint_name
    for checkpoint_name in model_last.pt "model_${target_update}.pt"; do
        PYTHONPATH=src "${PYTHON_BIN}" -u "${VALIDATOR}" \
            --checkpoint "${checkpoint_dir}/${checkpoint_name}" \
            --contract "${checkpoint_dir}/training_contract.json" \
            --stage "${stage}" \
            --update "${target_update}" \
            --horizon "${horizon}" \
            --expected-world-size "${NUM_PROCESSES}" \
            --expected-mixed-precision bf16 \
            --expected-frame-budget "${FRAME_BUDGET_PER_GPU}" \
            --expected-max-samples "${MAX_SAMPLES}"
    done
    if [[ "${stage}" == s2c ]]; then
        PYTHONPATH=src "${PYTHON_BIN}" -u "${VALIDATOR}" \
            --checkpoint "${checkpoint_dir}/model_20000.pt" \
            --contract "${checkpoint_dir}/training_contract.json" \
            --stage s2c \
            --update 20000 \
            --horizon 70000 \
            --expected-world-size "${NUM_PROCESSES}" \
            --expected-mixed-precision bf16 \
            --expected-frame-budget "${FRAME_BUDGET_PER_GPU}" \
            --expected-max-samples "${MAX_SAMPLES}"
    fi
}

cd "${PROJECT_ROOT}"
echo "Warm-start chain ${CHAIN_RUN_ID}: stages=${STAGES[*]} GPUs=${GPU_IDS} frame_budget=${FRAME_BUDGET_PER_GPU}"
for stage in "${STAGES[@]}"; do
    horizon=$(stage_horizon "${stage}")
    target_update=${horizon}
    stage_log="${LOG_DIR}/pretrain_semantic_vae_warmstart_${stage}_6xa40_${CHAIN_RUN_ID}.log"
    assert_target_gpus_free
    checkpoint_dir=$(stage_checkpoint_dir "${stage}")
    PYTHONPATH=src "${PYTHON_BIN}" -u "${VALIDATOR}" --resume-directory "${checkpoint_dir}"
    active_processes=$(active_warmstart_processes)
    if [[ -n "${active_processes}" ]]; then
        echo "Refusing to start ${stage}: another warm-start training process is active:" >&2
        printf '%s\n' "${active_processes}" >&2
        exit 4
    fi

    start_epoch=$(date +%s)
    start_timestamp=$(date '+%F %T %Z')
    record_event "${stage}" start "${start_timestamp}" "${start_epoch}" 0 "${target_update}" 0
    echo "[$(date '+%F %T %Z')] starting ${stage} -> update ${target_update}; log=${stage_log}"
    start_monitor "${stage}"
    ROOT_PREFIX="${ROOT_PREFIX}" \
        RUN_UNTIL_UPDATE="${target_update}" \
        GPU_IDS="${GPU_IDS}" \
        NUM_PROCESSES="${NUM_PROCESSES}" \
        FRAME_BUDGET_PER_GPU="${FRAME_BUDGET_PER_GPU}" \
        MAX_SAMPLES="${MAX_SAMPLES}" \
        MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT}" \
        OMP_NUM_THREADS=1 \
        NCCL_TIMEOUT=1200 \
        NCCL_IB_DISABLE=1 \
        NCCL_P2P_DISABLE=1 \
        NCCL_DEBUG=WARN \
        setsid bash "${STAGE_LAUNCHER}" "${stage}" >> "${stage_log}" 2>&1 &
    STAGE_PID=$!
    if wait "${STAGE_PID}"; then
        stage_exit_code=0
    else
        stage_exit_code=$?
    fi
    STAGE_PID=""
    stop_monitor
    end_epoch=$(date +%s)
    end_timestamp=$(date '+%F %T %Z')
    elapsed_seconds=$((end_epoch - start_epoch))
    record_event \
        "${stage}" end "${end_timestamp}" "${end_epoch}" "${elapsed_seconds}" \
        "${target_update}" "${stage_exit_code}"
    echo "[$(date '+%F %T %Z')] ${stage} launcher exit=${stage_exit_code}; elapsed=${elapsed_seconds}s"
    if (( stage_exit_code != 0 )); then
        echo "Stopping chain because ${stage} failed; inspect ${stage_log}" >&2
        exit "${stage_exit_code}"
    fi

    validate_stage "${stage}" "${target_update}" "${horizon}"
    echo "[$(date '+%F %T %Z')] ${stage} checkpoint validation passed"
    awk -F, -v st="${stage}" '
        $2 == st && $3 >= 2 && $3 <= 7 {
            gpu=$3+0; memory=$4+0
            if (memory > peak[gpu]) peak[gpu]=memory
        }
        END {
            printf "stage=%s sampled_peak_memory",st
            for (gpu=2;gpu<=7;gpu++) printf " GPU%d=%dMiB",gpu,peak[gpu]
            printf "\n"
        }
    ' "${METRICS_LOG}"
done

echo "[$(date '+%F %T %Z')] warm-start chain ${CHAIN_RUN_ID} completed all requested pure-audio stages"
