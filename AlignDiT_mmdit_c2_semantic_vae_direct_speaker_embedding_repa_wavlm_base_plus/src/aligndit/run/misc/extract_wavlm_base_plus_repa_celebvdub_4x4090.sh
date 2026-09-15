#!/bin/bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$script_dir"
while [[ "$project_root" != "/" && ! -f "$project_root/env.sh" ]]; do
    project_root="$(dirname "$project_root")"
done
if [[ ! -f "$project_root/env.sh" ]]; then
    echo "Cannot locate project env.sh from $script_dir" >&2
    exit 1
fi
# shellcheck source=/dev/null
source "$project_root/env.sh"
cd "$project_root"

python_bin="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
manifest="${ROOT_PREFIX}/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32/manifests/train.jsonl"
audio_root="${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/audio"
cache_dir="${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16"
for path in "$python_bin" "$manifest"; do
    if [[ ! -f "$path" || -L "$path" ]]; then
        echo "Missing required regular file: $path" >&2
        exit 1
    fi
done
if [[ ! -d "$audio_root" || -L "$audio_root" ]]; then
    echo "Missing regular CelebV-Dub audio directory: $audio_root" >&2
    exit 1
fi
if [[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" -lt 4 ]]; then
    echo "WavLM extraction requires four visible GPUs" >&2
    exit 1
fi

echo "Caching pinned final-layer WavLM-Base+ targets with four GPUs at $cache_dir" >&2
exec env \
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    OMP_NUM_THREADS=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=src \
    "$python_bin" -u -m torch.distributed.run \
        --standalone \
        --nproc_per_node=4 \
        src/aligndit/script/misc/extract_wavlm_base_plus_repa.py \
        --manifest "$manifest" \
        --audio-root "$audio_root" \
        --cache-dir "$cache_dir" \
        "$@"
