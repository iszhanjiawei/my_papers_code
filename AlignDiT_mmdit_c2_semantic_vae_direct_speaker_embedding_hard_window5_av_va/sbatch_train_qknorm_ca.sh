#!/bin/bash
# Historical launch entry retained only as source reference in this snapshot.
echo "This isolated snapshot trains only the five-frame hard AV+VA window." >&2
echo "Use src/aligndit/run/train/finetune_celebvdub_svae_speaker_hard_window5_av_va_4x4090.sh" >&2
exit 1

/share/slurm.pl --job-name=mmdit_prompt_stagewise --nodelist=xju-aslp8 --gpu 4 --num-threads 64 ./mmdit_prompt_stagewise.log \
    bash src/aligndit/run/train/finetune_celebvdub_mm_slurm.sh
