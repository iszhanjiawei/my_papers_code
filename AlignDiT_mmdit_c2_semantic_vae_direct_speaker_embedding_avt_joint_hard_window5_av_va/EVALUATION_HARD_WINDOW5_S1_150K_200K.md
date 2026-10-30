# Strict AVT hard-window5: 150k and 200k EMA evaluation

This report concerns only
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window5_av_va`.
The strict AV/VA window has radius 2 (five frames); all audio/video/text streams
retain joint attention. The evaluator and model source are unchanged from
commit `159a7e9f48c930690d5b3559d072173091d98494`.

## Checkpoints and protocol

Training finished at update 200000. The 150k numbered checkpoint was saved on
2026-10-29 at 23:06:54 +08:00 and the 200k numbered checkpoint on 2026-10-30
at 08:54:20 +08:00. The evaluated weights are EMA, with exact checkpoint update
and matching strict-five-frame training contract checked before inference.
Both historical checkpoints contain 802 online model state keys and an
initialized EMA. EMA's internal step is 220000 / 270000, following the inherited
70000-step parent policy; training updates remain 150000 / 200000.

Both evaluations use all 213 CelebV-Dub Setting 1 clips, seed 0, CFG text/video
5/2, Euler + EPSS, sway -1, 32 NFE and exact target duration. Reference prompt
and CAM++ speaker vector come from the same clip's complete GT audio. WAVs
contain only the target segment. The Semantic-VAE 1000k decoder, LibriSpeech
train latent normalization and CAM++ cache remain unchanged. These are the
same protocol settings used for the audited hard-window9 experiments.

The pipeline generates WAVs, computes SPKSIM (WavLM), WER
(faster-whisper-large-v3, English, beam 5, FP16) and EMOSIM
(emotion2vec-plus-large), extracts AV-HuBERT features from generated audio plus
the matching mouth video, then computes AVSync. Each metric uses one worker.

## Reproduction

Run from this project's root with available physical GPUs 0 and 1. The
launches use independent sessions, unbuffered logging, the existing Python
environment and this snapshot's package path. After each launch, confirm that
the session ID equals the recorded PID before the launching shell exits.

```bash
source env.sh
mkdir -p logs
for entry in 150000:0 200000:1; do
    step="${entry%:*}"
    gpu="${entry#*:}"
    log="logs/eval_s1_${step}_hard_window5_cfgv2.0_20261030.log"
    setsid env PYTHONUNBUFFERED=1 CKPT_STEP="$step" EVAL_GPU="$gpu" \
      CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
      MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
      > "$log" 2>&1 < /dev/null &
    eval_pid=$!
    printf '%s\n' "$eval_pid" > "${log%.log}.pid"
    for attempt in {1..100}; do
        actual_sid="$(ps -o sid= -p "$eval_pid" | tr -d ' ')"
        [[ "$actual_sid" == "$eval_pid" ]] && break
        sleep 0.1
    done
    [[ "$actual_sid" == "$eval_pid" ]] || exit 1
    ps -o pid,ppid,sid,tty,stat,cmd -p "$eval_pid"
done
```

Do not rerun inference into an existing nonempty output directory. The existing
inference entry refuses to overwrite results. Runtime launch records, stage
logs and audit logs are stored in ignored `logs/`; no runtime files are added
to Git.

Checkpoint/output root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window5_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Each run uses `model_<step>.pt` and writes to
`eval_s1_<step>_speaker_avt_joint_hard_window5_av_va_cfgv2.0` under that root.

After both pipelines finish, audit each output with:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/audit_avt_joint_s1.py \
  --output-dir "<checkpoint-root>/eval_s1_150000_speaker_avt_joint_hard_window5_av_va_cfgv2.0" \
  --output-dir "<checkpoint-root>/eval_s1_200000_speaker_avt_joint_hard_window5_av_va_cfgv2.0"
```

The audit checks the complete 213 unique full clip IDs, exact WAV and feature
sets, valid waveform samples and SHA256 receipts, finite features with GT
shapes and all four metric record counts. It recomputes sample means and
frame-cosine AVSync. Corpus WER is total word errors divided by reference word
count, rather than an average of utterance WER. Per-clip results are exported
with full IDs to preserve cross-video duplicate basenames.

## Results

Pending completion of both complete-set evaluations and independent audits.
