#!/bin/bash
# --- ROOT_PREFIX path switch (auto-load env.sh) ---
__envdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "$__envdir" != "/" ] && [ ! -f "$__envdir/env.sh" ]; do __envdir="$(dirname "$__envdir")"; done
[ -f "$__envdir/env.sh" ] && source "$__envdir/env.sh"
# --------------------------------------------------
# Extract Tacotron mel-spectrograms for CelebV-Dub (train split) using CPU sharding.

FILE=$(realpath "$0" | sed 's|/run/|/script/|g' | sed 's/\.sh$/.py/')
# point to the shared extract_mel implementation
FILE=$(dirname "$FILE")/extract_mel.py

NSHARD=32

trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT
for RANK in $(seq 0 $((NSHARD-1))); do
    OMP_NUM_THREADS=1 \
    PYTHONPATH=src \
    ${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u "$FILE" \
    --nshard ${NSHARD} \
    --rank ${RANK} \
    --input-dir "data/CelebVDub/audio/train" \
    --output-dir "data/CelebVDub/mel_tacotron/train" \
    --file-extension ".wav" \
    &
done
wait
echo "ALL MEL SHARDS DONE"
