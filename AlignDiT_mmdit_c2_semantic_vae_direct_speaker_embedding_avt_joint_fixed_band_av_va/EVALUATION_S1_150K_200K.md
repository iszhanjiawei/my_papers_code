# AVT joint + fixed AV/VA band: 150k / 200k evaluation

Evaluate the numbered 150,000 and 200,000 checkpoints of this snapshot, using
their EMA weights and the fixed-band inference contract. The original
CelebV-Dub Setting 1 protocol uses all 213 test clips, each clip's full GT
reference prompt and CAM++ speaker vector, real target duration, seed 0,
CFG text/video 5/2, Euler with EPSS, sway -1 and 32 NFE. The generated WAV
contains only the target segment. This is the same-clip reference protocol.

The existing evaluation entry generates audio, computes SPKSIM (WavLM), WER
(faster-whisper-large-v3) and EMOSIM (emotion2vec_plus_large), then extracts
AV-HuBERT features from the generated audio and corresponding mouth video
and computes AVSync. Each checkpoint has a separate output directory.

Run from this project's root on two available GPUs:

```bash
source env.sh
mkdir -p logs
for checkpoint_step in 150000 200000; do
  eval_gpu=0
  if [[ "$checkpoint_step" == 200000 ]]; then eval_gpu=1; fi
  setsid env PYTHONUNBUFFERED=1 CKPT_STEP="$checkpoint_step" EVAL_GPU="$eval_gpu" \
    CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
    bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
    > "logs/eval_s1_${checkpoint_step}_fixed_band_cfgv2.0.log" 2>&1 < /dev/null &
  eval_pid=$!
  printf '%s\n' "$eval_pid" > "logs/eval_s1_${checkpoint_step}_fixed_band_cfgv2.0.pid"
  ps -o pid,ppid,sid,tty,stat,cmd -p "$eval_pid"
done
```

After both pipelines finish, audit each output using
`PYTHONPATH=src ${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python -u scripts/audit_avt_joint_s1.py --output-dir <output>`.
The audit requires exactly 213 WAVs, 213 AV-HuBERT features and 213 records per
metric; checks full clip identities, audio checksums and duration receipts;
and recomputes the reported metrics. WER is total word edits divided by total
reference words, not an average of utterance WER. Output artifacts remain
outside Git, alongside the checkpoints.

## Results

Pending completion and coverage audit.
