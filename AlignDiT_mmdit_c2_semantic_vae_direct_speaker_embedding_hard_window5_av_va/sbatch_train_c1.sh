#!/bin/bash
# Historical launch entry retained only as source reference in this snapshot.
echo "This isolated snapshot trains only the five-frame hard AV+VA window." >&2
echo "Use src/aligndit/run/train/finetune_celebvdub_svae_speaker_hard_window5_av_va_4x4090.sh" >&2
exit 1

/share/slurm.pl --job-name=mmdit_c1_prompt_tailtext --nodelist=xju-aslp8 --gpu 4 --num-threads 64 \
    ./mmdit_c1_prompt_tailtext.log \
    bash src/aligndit/run/train/finetune_celebvdub_mm_c1_slurm.sh
