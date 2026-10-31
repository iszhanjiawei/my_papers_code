> Historical source experiment record copied for provenance. It does not report WavLM TRA training or evaluation. See [the active experiment](WAVLM_TRA_HARD_WINDOW9.md).

# Strict AVT hard-window9: 200k EMA evaluation

Evaluate this experiment's completed `model_200000.pt`, retaining strict
bidirectional AV/VA radius 4 (nine frames). Training completed at update
200,000 on 2026-10-28; the numbered checkpoint and final checkpoint were
saved around 16:41 Asia/Shanghai and the training launcher exited.

Use the same protocol as the audited 150k evaluation: all 213 CelebV-Dub
Setting 1 test clips, EMA, seed 0, CFG text/video 5/2, Euler + EPSS,
sway -1, 32 NFE and real target duration. The prompt and CAM++ speaker
vector come from the same clip's full GT audio. Generated WAVs contain
only the target segment. Decoder, latent normalization and speaker cache
follow the unchanged experiment configuration.

Run from this project root on available physical GPU 1:

```bash
source env.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=200000 EVAL_GPU=1 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_200000_hard_window9_cfgv2.0_20261028.log 2>&1 < /dev/null &
eval_pid=$!
printf '%s\n' "$eval_pid" > logs/eval_s1_200000_hard_window9_cfgv2.0_20261028.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$eval_pid"
```

The existing single-GPU pipeline generates WAVs, computes SPKSIM (WavLM),
WER (faster-whisper-large-v3) and EMOSIM (emotion2vec-plus-large), extracts
AV-HuBERT features from generated audio and corresponding mouth video,
then computes AVSync. No model or evaluator source changes are needed.

Checkpoint/output root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Output subdirectory: `eval_s1_200000_speaker_avt_joint_hard_window9_av_va_cfgv2.0`.

After successful pipeline completion:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u scripts/audit_avt_joint_s1.py \
  --output-dir "${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_200000_speaker_avt_joint_hard_window9_av_va_cfgv2.0" \
  > logs/audit_s1_200000_hard_window9_20261028.log 2>&1
```

The audit requires exactly 213 WAVs, 213 generated AV-HuBERT features and
213 records in each metric file. It checks full clip identities, finite
values, WAV checksums/sample counts, feature shapes and recomputed metric
values. WER uses total word edit counts / total reference words. Runtime
artifacts remain outside Git.

## Results

Completed on 2026-10-28 on physical GPU 1. All four metrics and the independent
coverage/metric audit pass for **213/213** clips.

| EMA checkpoint | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| Strict hard-window9, 200k | 0.65990 | 0.04752 | 0.76125 | 0.56899 |

Corpus WER is **113 word errors / 2,378 reference words = 0.0475189235**,
or 4.75189%.

Checkpoint SHA256:
`a80a8395f3c65c1f8fd376307594335c024ab8cfe539d5e7bb3f0c510aa532b0`.

The audit verifies exactly 213 WAVs, 213 generated AV-HuBERT features and
213 records per metric, excluding blank lines and summary lines. WAVs are
finite 16-kHz mono, have the inference receipt's sample count and match its
SHA256. Generated feature shapes match GT and all values are finite. Full
test/video/clip identities are retained, including clips with identical
basenames in different videos. AVSync is recomputed from the feature arrays;
the other sample means and corpus WER are recomputed from per-clip records.
All recomputed values agree with the original evaluator's reported results.

## Same-run 150k / 200k comparison

The inference summaries match in generation settings, test-list and latent
normalization hashes, decoder metadata and same-clip speaker/reference sources.
Both checkpoints come from this same strict-window9 training run and both
audits cover the complete 213-clip test set.

| EMA checkpoint | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| 150k | 0.65257 | 0.04542 | 0.78132 | 0.57165 |
| 200k | 0.65990 | 0.04752 | 0.76125 | 0.56899 |

200k has higher SPKSIM by 0.00733. 150k has lower WER (108 vs 113 word errors),
higher EMOSIM and higher AVSync. This is a descriptive comparison on the same
test set; the four metrics favor different checkpoints.

Each checkpoint's output directory retains its target WAVs under `test/`,
generated features under `avhubert_feat/test/`, `inference_summary.json`,
four original `_*_results.jsonl` files, full-ID `evaluation_per_clip.jsonl`
and `evaluation_audit.json`. The 200k directory additionally contains
`comparison_protocol_150k_200k.json` and `comparison_with_150k.json`.

The project's ignored run record is
`logs/evaluation_hard_window9_s1_200k_20261028.json`. Original stage and audit
logs use the filenames in the reproduction commands above. Model,
configuration and evaluator source remain those of the existing experiment;
this task adds only the reproducibility/results document to Git.
