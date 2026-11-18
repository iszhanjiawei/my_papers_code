#!/bin/bash
/share/slurm.pl --job-name=mmdit_c2_global_stage --nodelist=xju-aslp8 --gpu 4 --num-threads 64 \
    ./mmdit_c2_global_stage.log \
    bash src/aligndit/run/train/finetune_celebvdub_mm_c2_slurm.sh
