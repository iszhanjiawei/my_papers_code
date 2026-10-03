#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/../../../.." && pwd)"
source "$project_root/env.sh"
cd "$project_root"
mkdir -p output
exec 9>output/chem_training.lock
if ! flock -n 9; then
    echo "Another Chem training launcher already holds the project lock" >&2
    exit 1
fi
python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
if [[ ! -x "$python_bin" ]]; then
    echo "Missing aligndit Python: $python_bin" >&2
    exit 1
fi
# Shared GPUs are permitted; the caller selects a device after checking memory.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    echo "Chem exact-resume training requires one selected GPU" >&2
    exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root/src${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
exec "$python_bin" -u -m accelerate.commands.launch \
    --mixed_precision bf16 --num_machines 1 --num_processes 1 \
    --dynamo_backend no --main_process_port "${TRAIN_PORT:-29644}" \
    src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker.py \
    --config-name finetune_chem_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus "$@"
