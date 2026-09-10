#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source env.sh
PY="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
DATA="${ROOT_PREFIX}/zjw524/projects/data"
BENCH="${REPEAT213_DATASET:-$DATA/evaluations/celebvdub_train_repeat213_seed666_en_20260910}"
export CUDA_VISIBLE_DEVICES="${EVAL_GPU:-0}" OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONPATH=src
export PATH="$(dirname "$PY"):$PATH"
# Initialize once to avoid concurrent downloads of the same pretrained FAN assets.
"$PY" -c "import face_alignment; face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device='cuda')"
pids=()
for rank in ${CROP_RANKS:-0 1 2 3}; do
    "$PY" -u scripts/prepare_repeat213_mouth.py --dataset "$BENCH" \
        --mean-face "$DATA/pretrained_models/mouth_crop/20words_mean_face.npy" --rank "$rank" --nshard 4 &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
exit "$status"
