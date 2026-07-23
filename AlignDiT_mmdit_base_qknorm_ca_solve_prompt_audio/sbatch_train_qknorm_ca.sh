#!/bin/bash
/share/slurm.pl --job-name=mmdit_qknorm_ca --nodelist=xju-aslp8 --gpu 4 --num-threads 64 ./mmdit_qknorm_ca.log \
    bash src/aligndit/run/train/finetune_celebvdub_mm_slurm.sh
