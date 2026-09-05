#!/usr/bin/env bash
set -euo pipefail

# Frozen CAM++ preprocessing for the exact 79,613 training Arrow rows plus the
# 213 CelebV-Dub Setting-1 prompts. Embeddings come from complete waveforms.
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
if [[ ! -f "$__envdir/env.sh" ]]; then
    echo "Cannot locate this experiment's env.sh" >&2
    exit 1
fi
source "$__envdir/env.sh"
cd "$__envdir"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
OMP_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
PYTHONPATH=src \
"${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u -m torch.distributed.run \
    --standalone \
    --nproc-per-node="${NUM_PROCESSES:-4}" \
    src/aligndit/script/misc/extract_campplus_speaker_embeddings.py \
    --dataset-arrow "${ROOT_PREFIX}/zjw524/projects/data/CelebVDub_char/raw.arrow" \
    --data-dir "${ROOT_PREFIX}/zjw524/projects/data" \
    --audio-root "${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/audio" \
    --cache-dir "${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/campplus_spk_emb_zh_en_16k" \
    --checkpoint "${ROOT_PREFIX}/zjw524/projects/data/pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt" \
    --test-list "${ROOT_PREFIX}/zjw524/projects/data/celebvdub_test_s1.lst" \
    --batch-utterances 32 \
    --num-workers 8
