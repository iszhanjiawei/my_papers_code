#!/bin/bash
/share/slurm.pl --job-name=mmdit_c1_prompt_tailtext --nodelist=xju-aslp8 --gpu 4 --num-threads 64 \
    ./mmdit_c1_prompt_tailtext.log \
    bash src/aligndit/run/train/finetune_celebvdub_mm_c1_slurm.sh
