> Historical record copied from the hard-window9 source experiment.
> For this snapshot, use [AVT_HARD_WINDOW5_AV_VA.md](AVT_HARD_WINDOW5_AV_VA.md); these results and commands describe the source run.

# Strict AVT hard-window9: 150k EMA evaluation

Target: this experiment's `model_150000.pt`, with the original strict AV/VA
radius 4 (nine frames) training/inference contract. Evaluate all 213
CelebV-Dub Setting 1 test clips with EMA, seed 0, CFG text/video 5/2,
Euler + EPSS, sway -1, 32 NFE and real target duration. The prompt and CAM++
speaker vector come from the same clip's full GT audio, following the
existing same-clip reference protocol. Generated WAVs contain only the
target segment.

Inference and the four metrics reuse the existing single-GPU entry. Physical
GPU 1 shares with the ongoing four-GPU training; inference/evaluation run
with one worker and keep the original model settings. Monitor GPU memory,
training and evaluation progress until completion.

Run from this project root:

```bash
source env.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=150000 EVAL_GPU=1 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_150000_hard_window9_cfgv2.0_20261028.log 2>&1 < /dev/null &
eval_pid=$!
printf '%s\n' "$eval_pid" > logs/eval_s1_150000_hard_window9_cfgv2.0_20261028.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$eval_pid"
```

The pipeline generates WAVs, computes SPKSIM (WavLM), WER
(faster-whisper-large-v3) and EMOSIM (emotion2vec-plus-large), extracts
AV-HuBERT features from generated audio plus the matching mouth video,
then computes AVSync. The decoder, latent normalization, speaker cache and
test protocol follow the existing experiment configuration.

Checkpoint/output root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Output subdirectory: `eval_s1_150000_speaker_avt_joint_hard_window9_av_va_cfgv2.0`.

After the pipeline finishes, audit using the standard interpreter and local
source snapshot:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/audit_avt_joint_s1.py \
  --output-dir "${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_150000_speaker_avt_joint_hard_window9_av_va_cfgv2.0" \
  > logs/audit_s1_150000_hard_window9_20261028.log 2>&1
```

The audit requires exactly 213 WAVs/features and records in each metric,
checks complete clip identities, finite values, WAV hashes and sample counts,
and recomputes the four reported metrics. WER uses total word edit counts
divided by total reference words. Runtime artifacts stay outside Git.

## Results

Completed on 2026-10-28 on physical GPU 1, while the original training
continued. The pipeline and independent audit both completed successfully.

| EMA checkpoint | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| Strict hard-window9, 150k | 0.65257 | 0.04542 | 0.78132 | 0.57165 |

All four metrics cover **213/213** clips. Corpus WER is **108 word errors /
2,378 reference words = 0.0454163162**, or 4.54163%.

Checkpoint SHA256:
`753ec3fed33489bf806234011a996bbb98a402bc31ffe26986a57713020c70c5`.

The audit verifies exactly 213 generated WAVs, 213 AV-HuBERT features and
213 records per metric. Every WAV is finite 16-kHz mono, matches its recorded
sample count and SHA256, and maps to the complete test-list clip ID. Feature
shapes match GT and values are finite. AVSync is recomputed from the arrays;
SPKSIM/EMOSIM means and corpus WER are recomputed from per-clip records. All
recomputed values agree with the original evaluator's reported values.
Blank lines and summary lines are excluded from sample counts.

## Existing Gaussian 150k comparison

The existing Gaussian AVT 150k result uses the same 213 clips, generation
settings, normalization, decoder, same-clip reference and speaker metadata;
these were compared directly between both inference summaries. Its audited
result is retained in the sibling Gaussian experiment's output directory.

| EMA 150k experiment | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| Gaussian AVT fixed-band | 0.64895 | 0.04668 | 0.74785 | 0.57954 |
| Strict AVT hard-window9 | 0.65257 | 0.04542 | 0.78132 | 0.57165 |

Hard-window9 has higher SPKSIM and EMOSIM, and three fewer word errors
(108 vs 111). Its AVSync is lower by 0.00788. This is a descriptive comparison
of independently trained checkpoints, without a statistical significance
or universal-improvement claim.

The output directory contains target WAVs under `test/`, generated features
under `avhubert_feat/test/`, `inference_summary.json`, all four original
`_*_results.jsonl` files, full-ID `evaluation_per_clip.jsonl`,
`evaluation_audit.json` and `comparison_with_gaussian_150k.json`.

The project's ignored run record is
`logs/evaluation_hard_window9_s1_150k_20261028.json`. Original stage logs and
the audit log use the filenames in the reproduction commands above. Only
this reproducibility/results document changes in Git; evaluation/model
source, configuration and generated artifacts are retained separately.
