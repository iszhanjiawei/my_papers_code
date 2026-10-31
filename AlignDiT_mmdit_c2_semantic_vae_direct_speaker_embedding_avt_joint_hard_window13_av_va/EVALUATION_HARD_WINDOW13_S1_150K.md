# Strict AVT hard-window13: 150k EMA evaluation

This report concerns only the independent
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window13_av_va`
snapshot. AV and VA attention use a strict radius of six aligned 40-Hz frames:
thirteen permitted keys for interior queries, clipped at sequence boundaries.
Outside-window direct AV/VA probabilities are zero. Audio, video and text retain
joint attention; the architecture remains 12 MM-DiT and 6 audio-only blocks.

## Checkpoint and protocol

Training stopped at update **150000** on 2026-10-31. The numbered checkpoint
was saved at **21:43:20 +08:00**; the final last checkpoint was saved at
21:43:50. This evaluation loads EMA from `model_150000.pt`, with the exact
checkpoint update and matching hard13 training contract checked before
inference. There are 802 online model state keys, an initialized EMA with
804 state keys, and EMA internal step 220000, including the inherited 70k
parent counter.

Checkpoint SHA256:
`218ec8d59271511dc3ca76f55fc50fca26e2a8946203c0946def3eb29cd62413`.

The protocol uses all **213 CelebV-Dub Setting 1 clips**, seed 0, CFG text/video
**5/2**, Euler + EPSS, sway -1 and **32 NFE**. Prompt and CAM++ speaker vector
come from the same clip's full GT audio. Generated WAVs contain only the target
segment, trimmed to its original sample count. The Semantic-VAE 1000k decoder,
64-D/40-Hz latents, LibriSpeech-train normalization and speaker cache retain the
previous hard5/hard9 settings.

The existing pipeline generates WAVs, computes SPKSIM with WavLM, WER with
English faster-whisper-large-v3 (beam 5, FP16), and EMOSIM with
emotion2vec-plus-large. It then extracts AV-HuBERT features from generated
audio and the matching mouth video and computes AVSync. Each metric uses
one worker. This run uses physical **GPU 0**.

## Reproduction and artifacts

Run from this snapshot's root. The existing evaluation scripts select the
hard13 configuration, checkpoint and output directory without code changes.
The launch uses the existing environment and this snapshot's `PYTHONPATH`.

```bash
source env.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=150000 EVAL_GPU=0 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_150000_hard_window13_cfgv2.0_20261031.log 2>&1 < /dev/null &
eval_pid=$!
printf '%s\n' "$eval_pid" > logs/eval_s1_150000_hard_window13_cfgv2.0_20261031.pid
```

Keep the launching shell alive until `ps` confirms `SID=PID`, `TTY=?` before
exiting it. Actual detached pipeline PID/SID: **3497411**. Do not rerun
inference into a nonempty output directory; the inference launcher refuses
to overwrite existing results.

Checkpoint root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window13_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Output directory under that root:
`eval_s1_150000_speaker_avt_joint_hard_window13_av_va_cfgv2.0`.

After all pipeline stages finish, independently audit the output:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/audit_avt_joint_s1.py \
  --output-dir "<checkpoint-root>/eval_s1_150000_speaker_avt_joint_hard_window13_av_va_cfgv2.0"
```

The audit checks all 213 full clip IDs, exact WAV and feature sets, waveform
SHA256 receipts, valid finite arrays, feature shapes against GT and four
complete metric records. It recomputes mean SPKSIM/EMOSIM/AVSync and corpus
WER from total word edit counts divided by total reference words. JSONL
summary lines are excluded from sample counts. The ordered basename-only
metric records are checked against the full test list before exporting
`evaluation_per_clip.jsonl`, preserving duplicate basenames across videos.

## Results

The pipeline finished successfully on **2026-10-31 at 22:22:23 +08:00** and
the independent audit passed. Additional waveform/protocol checks passed at
22:23:11. All four metrics cover **213/213** clips. The inference/evaluation
source is unchanged from revision
`68e096f0a242e93513e9274dba41e624d3d8713a`.

| EMA checkpoint | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| hard13, 150k | 0.65136 | 0.05257 | 0.75859 | 0.56432 |

Corpus WER is **125 word errors / 2378 reference words = 0.0525651808**
(5.25652%). The displayed values are independently recomputed from the full
per-clip records, with agreement to the evaluator summaries at five decimals.

Every generated WAV is finite, nonempty, 16-kHz mono PCM16 and has exactly
the original target sample count in the Semantic-VAE test manifest. The exact
WAV/feature sets and waveform SHA256 receipts match all full test IDs.
Generated AV-HuBERT feature shapes match GT; per-clip AVSync was independently
recomputed. There are exactly 213 valid records in each metric file.

## Matched-protocol 150k comparison

The historical hard5 and hard9 reports below are read-only references to
their completed independent runs. The complete `generation`, `dataset`,
`decoder`, `speaker_embedding` and `normalization_contract` metadata in the
three inference summaries are identical. This is a descriptive comparison
of the available independently trained checkpoints on the same protocol.

| Strict window, 150k EMA | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| 5 frames | 0.65115 | 0.05088 | 0.74839 | 0.56307 |
| 9 frames | 0.65257 | 0.04542 | 0.78132 | 0.57165 |
| 13 frames | 0.65136 | 0.05257 | 0.75859 | 0.56432 |

Compared with hard5, hard13 has SPKSIM +0.00021, EMOSIM +0.01020 and
AVSync +0.00125; WER is higher by 0.00168 (four additional word errors).
Compared with hard9, hard13 has SPKSIM −0.00121, EMOSIM −0.02273 and
AVSync −0.00734; WER is higher by 0.00715 (17 additional word errors).
The hard9 checkpoint has the best observed value of all four metrics among
these three 150k evaluations.

## Saved evidence

The output directory contains:

- `test/`: 213 generated target WAVs;
- `avhubert_feat/test/`: 213 generated AV-HuBERT features;
- `inference_summary.json`: EMA checkpoint SHA256, generation, decoder,
  normalization, reference/speaker metadata and per-WAV receipts;
- `_sim_results.jsonl`, `_wer_results.jsonl`, `_emosim_results.jsonl`,
  `_avsync_results.jsonl`: original metric records and summaries;
- `evaluation_per_clip.jsonl`: metrics with full clip IDs and word counts;
- `evaluation_audit.json`: complete coverage and recomputed aggregates;
- `comparison_protocol_hard5_hard9_hard13_150k.json`: exact protocol equality,
  target WAV format/sample-count validation and descriptive comparisons.

Runtime project records are excluded from Git:

- `logs/eval_s1_150000_hard_window13_cfgv2.0_20261031.log` and its PID/start files;
- `logs/evaluation_checkpoint_hard_window13_150k.json`;
- `logs/audit_s1_150k_hard_window13_20261031.log`;
- `logs/comparison_s1_150k_hard_window13_20261031.log`;
- `logs/evaluation_hard_window13_s1_150k_20261031.json`.

The detached pipeline and its workers have exited. The model, training
configuration, inference/evaluation code and historical experiment outputs
were not modified during evaluation; only this report and its documentation
link are committed.
