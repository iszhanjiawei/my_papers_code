#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
num_gpus="${NUM_PROCESSES:-4}"
mkdir -p logs
exec /s7home/slurm.pl --nodelist=xju-aslp8 --nodes=1 --ntasks=1 \
    --gpu "$num_gpus" --num-threads 64 --job-name=libritts_foley_grid \
    "./logs/grid_$(date +%Y%m%d_%H%M%S).log" \
    bash src/aligndit/run/train/finetune_grid.sh "$@"
