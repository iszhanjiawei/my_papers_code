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

Completed on **2026-10-30**, physical GPU **0** for 150k and GPU **1** for 200k.
Both pipelines finished successfully and both independent audits pass for
**213/213** clips. All metric values below are recomputed from the complete
per-clip records; summary lines are excluded.

| EMA checkpoint | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| 150k | 0.65115 | 0.05088 | 0.74839 | 0.56307 |
| 200k | 0.65686 | 0.05719 | 0.73519 | 0.56215 |

Corpus WER uses **2,378 reference words** in each evaluation:
- 150k: **121 word errors / 2378 = 0.0508830950** (5.08831%).
- 200k: **136 word errors / 2378 = 0.0571909167** (5.71909%).

200k increases SPKSIM by 0.00571. 150k has lower WER
(121 versus 136 word errors), higher EMOSIM by 0.01320,
and higher AVSync by 0.00092. Thus the three latter metrics
favor 150k, while speaker similarity favors 200k. These are descriptive
checkpoint comparisons on the same test set.

## Integrity and matching protocols

Both runs have exactly 213 WAVs, 213 generated AV-HuBERT features and 213
records per metric. The audit checks full clip identities and ordered metric
mapping, waveform SHA256 and sample counts, finite arrays, generated feature
shapes against GT, feature-based AVSync and agreement with reported metrics.
An additional check confirms every WAV is 16-kHz mono PCM16 and has exactly
the original target sample count from the Semantic-VAE test manifest.

The complete generation, dataset, decoder, speaker-conditioning and latent
normalization metadata match between checkpoints. These fields also match
the audited hard9 evaluation protocol; no extra result is inferred from that
protocol check. Both checkpoints retain strict five-frame attention, with
the matching training sidecar required by inference.

Checkpoint SHA256:

- 150k: `ed7b0619d294d1e627d1828fa7a1ec864f509c568043d83e0ef044b2ef98ecb4`.
- 200k: `684f5c494e15e0fb1aa9f085ee28bca601c6c56e754ee92bf019fc61f3bfa0bc`.

## Artifacts

Each output directory listed above contains:

- `test/`: all 213 target WAVs;
- `avhubert_feat/test/`: all 213 generated features;
- `inference_summary.json`: generation/decoder/reference metadata and WAV receipts;
- `_sim_results.jsonl`, `_wer_results.jsonl`, `_emosim_results.jsonl`, `_avsync_results.jsonl`;
- `evaluation_per_clip.jsonl`: full-ID per-clip metrics and word-edit counts;
- `evaluation_audit.json`: complete coverage and recomputed aggregates.

The 200k directory additionally contains `comparison_protocol_150k_200k.json`
and `comparison_with_150k.json`.

Ignored project logs:

- `logs/eval_s1_150000_hard_window5_cfgv2.0_20261030.log`;
- `logs/eval_s1_200000_hard_window5_cfgv2.0_20261030.log`;
- `logs/audit_s1_150k_200k_hard_window5_20261030.log`;
- `logs/comparison_s1_150k_200k_hard_window5_20261030.log`;
- `logs/evaluation_hard_window5_s1_150k_200k_20261030.json`.

Actual detached evaluation pipeline PIDs were 1929593 (150k, GPU 0) and
1929612 (200k, GPU 1); both have exited successfully. Model, training, inference
and evaluator source remain unchanged. Only this reproduction/results document
is added to Git for this request.
