#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)
# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

# env.sh intentionally defaults to the direct /zjw524 layout. Some servers
# expose the same workspace below /home or /s7home; discover that prefix only
# when the caller has not explicitly selected one.
if [[ -z "${ROOT_PREFIX}" && ! -x "/zjw524/ENTER/envs/aligndit/bin/accelerate" ]]; then
    for candidate_prefix in /home /s7home; do
        if [[ -x "${candidate_prefix}/zjw524/ENTER/envs/aligndit/bin/accelerate" ]]; then
            export ROOT_PREFIX="${candidate_prefix}"
            break
        fi
    done
fi
ACCELERATE_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/accelerate"
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
    echo "AlignDiT accelerate executable does not exist: ${ACCELERATE_BIN}" >&2
    echo "Set ROOT_PREFIX to the prefix containing /zjw524 (for example /home)." >&2
    exit 2
fi

STAGE=${1:-${WARMSTART_STAGE:-s1}}
case "${STAGE}" in
    s1|s2a|s2b)
        DEFAULT_RUN_UNTIL=10000
        ;;
    s2c)
        # First stop at cumulative 50k (10k + 10k + 10k + 20k).
        # Resume the same S2c contract with RUN_UNTIL_UPDATE=70000 after
        # the dev gate passes; its LR scheduler is planned for all 70k.
        DEFAULT_RUN_UNTIL=20000
        ;;
    *)
        echo "Stage must be one of s1, s2a, s2b, s2c; got: ${STAGE}" >&2
        exit 2
        ;;
esac

GPU_IDS=${GPU_IDS:-2,3,4,5,6,7}
NUM_PROCESSES=${NUM_PROCESSES:-6}
MAIN_PROCESS_PORT=${MAIN_PROCESS_PORT:-29630}
FRAME_BUDGET_PER_GPU=${FRAME_BUDGET_PER_GPU:-7200}
MAX_SAMPLES=${MAX_SAMPLES:-32}
RUN_UNTIL_UPDATE=${RUN_UNTIL_UPDATE:-${DEFAULT_RUN_UNTIL}}

for value_name in NUM_PROCESSES FRAME_BUDGET_PER_GPU MAX_SAMPLES RUN_UNTIL_UPDATE; do
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
declare -A SEEN_GPUS=()
for gpu_id in "${GPU_ARRAY[@]}"; do
    if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]] || [[ -n "${SEEN_GPUS[${gpu_id}]:-}" ]]; then
        echo "GPU_IDS must contain unique non-negative integers, got: ${GPU_IDS}" >&2
        exit 2
    fi
    SEEN_GPUS[${gpu_id}]=1
done

ACCELERATE_DISTRIBUTED_ARGS=(--num_machines 1)
if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_DISTRIBUTED_ARGS+=(--multi_gpu --gpu_ids all)
fi

cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
export EXPECTED_WORLD_SIZE="${NUM_PROCESSES}"
export WARMSTART_RUN_UNTIL_UPDATE="${RUN_UNTIL_UPDATE}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1200}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONUNBUFFERED=1
export PYTHONPATH=src

# This A40 server's validated topology requires P2P/IB to be disabled; leaving
# either path enabled hangs during DDP parameter synchronization.  A different
# machine may explicitly override both values to 0 after its own NCCL canary.
exec "${ACCELERATE_BIN}" launch \
    "${ACCELERATE_DISTRIBUTED_ARGS[@]}" \
    --mixed_precision bf16 \
    --num_processes "${NUM_PROCESSES}" \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    src/aligndit/script/train/pretrain_semantic_vae_warmstart.py \
    --config-name "pretrain_semantic_vae_warmstart_${STAGE}" \
    datasets.batch_size_per_gpu="${FRAME_BUDGET_PER_GPU}" \
    datasets.max_samples="${MAX_SAMPLES}"
