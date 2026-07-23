#!/bin/bash
/share/slurm.pl --job-name=mmdit_base --nodelist=xju-aslp8 --gpu 4 --num-threads 64 ./mmdit_base.log \
    bash src/aligndit/run/train/finetune_celebvdub_mm_slurm.sh
