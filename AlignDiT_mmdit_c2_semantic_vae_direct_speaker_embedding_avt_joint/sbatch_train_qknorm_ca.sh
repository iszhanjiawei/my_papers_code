#!/bin/bash
/share/slurm.pl --job-name=mmdit_prompt_stagewise --nodelist=xju-aslp8 --gpu 4 --num-threads 64 ./mmdit_prompt_stagewise.log \
    bash src/aligndit/run/train/finetune_celebvdub_mm_slurm.sh
