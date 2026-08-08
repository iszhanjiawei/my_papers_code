#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)

# shellcheck source=/dev/null
source "${PROJECT_ROOT}/env.sh"

PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
CACHE_ROOT="${ROOT_PREFIX}/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32"
MANIFEST="${CACHE_ROOT}/manifests/inventory.jsonl"
DATASET_ROOT="${ROOT_PREFIX}/zjw524/datasets/CelebV-Dub"
CHECKPOINT_ROOT="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k"
SEMANTIC_REPO="${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE"
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
ATTEMPT_ID=${SVAECACHE_ATTEMPT_ID:-"$(date -u +%Y%m%dT%H%M%SZ)-$$"}
MANIFEST_SHA256=${CELEBVDUB_SVAE_MANIFEST_SHA256:-a6478cce785748cbcefd87af54eafa9f654d735afa1c41b8f846e041cbc1286d}

if [[ ! -x "${PYTHON}" ]]; then
    echo "AlignDiT Python is not executable: ${PYTHON}" >&2
    exit 1
fi
if [[ "${NPROC_PER_NODE}" != "4" ]]; then
    echo "The formal only-VAE extraction contract requires NPROC_PER_NODE=4, got ${NPROC_PER_NODE}" >&2
    exit 1
fi
if [[ ! "${MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "Export CELEBVDUB_SVAE_MANIFEST_SHA256 with the immutable inventory SHA256" >&2
    exit 1
fi
cd "${PROJECT_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONPATH=src
export PYTHONUNBUFFERED=1

echo "Starting CelebV-Dub Semantic-VAE extraction: attempt=${ATTEMPT_ID}, nproc=${NPROC_PER_NODE}"
exec "${PYTHON}" -m torch.distributed.run \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    src/aligndit/script/misc/extract_celebvdub_svae_latents.py \
    --dataset-root "${DATASET_ROOT}" \
    --cache-root "${CACHE_ROOT}" \
    --manifest "${MANIFEST}" \
    --checkpoint-root "${CHECKPOINT_ROOT}" \
    --semantic-vae-repo "${SEMANTIC_REPO}" \
    --attempt-id "${ATTEMPT_ID}" \
    --expected-manifest-sha256 "${MANIFEST_SHA256}" \
    "$@"
