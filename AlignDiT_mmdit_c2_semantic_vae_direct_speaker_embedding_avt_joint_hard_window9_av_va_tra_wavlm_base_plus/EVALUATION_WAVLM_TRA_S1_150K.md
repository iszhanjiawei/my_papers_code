# WavLM TRA + AVT hard-window9 — audited 150k EMA evaluation

Evaluated on 2026-11-02 using GPU 0 (RTX 4090). All stages completed and the final audit passed **213/213** clips. This document reports the new TRA experiment; earlier copied hard-window9 evaluation documents describe its source baseline.

## Results and matched baseline

| Experiment | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|
| Hard-window9 AVT, 150k (baseline) | 0.65257 | 0.04542 | 0.78132 | 0.57165 |
| Hard-window9 AVT + WavLM TRA, 150k | 0.66435 | 0.05172 | 0.75642 | 0.56554 |
| TRA minus baseline | +0.01178 | +0.00631 | -0.02490 | -0.00612 |

TRA corpus WER is **123/2378 = 0.051724137931 (5.17241%)**. The baseline has 108/2378 errors: TRA makes **15 more word errors**, an increase of 0.63078 percentage points.

At this checkpoint, speaker similarity improves while WER, emotion similarity and AVSync are worse than the original hard-window9 150k baseline. These are descriptive results from independent training runs and one fixed inference seed; no statistical significance or broad quality improvement is claimed.

## Protocol

- CelebV-Dub **Setting 1**, the complete **213** clips; exact test-list SHA256 `1ad609deefaf1293e00b168a3418611ef779c4e9b49f3b51539721a8360a1e49`.
- Checkpoint update **150000**, **EMA**, strict loading of all **808** model state tensors and validation of the embedded TRA identity, original training sidecar and strict nine-frame AV/VA contract.
- Euler + EPSS, **32 NFE**, seed **0**, sway **−1**, CFG text/video **5.0/2.0**. Original clip duration is preserved; prompt audio is removed from output.
- Historical same-clip **full GT audio prompt and CAM++ speaker reference**. This is the same evaluation condition used for the source baseline.
- Same Semantic-VAE decoder, LibriSpeech latent normalization and speaker metadata as baseline; exact equality is recorded in `comparison_protocol_with_hard9_150k.json`.
- SPKSIM uses the existing WavLM-large speaker evaluator, WER uses faster-whisper-large-v3, and EMOSIM uses emotion2vec-plus-large. The training teacher remains frozen WavLM-Base+; it is not used during synthesis.
- AVSync uses generated audio + matching GT mouth video jointly encoded by the existing AV-HuBERT checkpoint, then frame cosine similarity against the GT AV-HuBERT features and equal-weight sample averaging.
- Evaluation used one metric worker (`-n 1`). WER is total word edits / total reference words, not mean per-clip WER. Other three metrics are equal-weight sample means.

The generated-output protocol, test-list hash, normalization hash, decoder metadata, speaker metadata and normalization contract match the baseline. Method/implementation details: [WAVLM_TRA_HARD_WINDOW9.md](WAVLM_TRA_HARD_WINDOW9.md).

## Checkpoint and outputs

Checkpoint: `/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char/model_150000.pt`

Checkpoint SHA256: `f63394041a05a397ffa2ef041a7e984c5cb4305a5eab32209dec171782dc0251`

Baseline checkpoint SHA256: `753ec3fed33489bf806234011a996bbb98a402bc31ffe26986a57713020c70c5`

Output directory: `/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_150000_speaker_avt_joint_hard_window9_av_va_tra_wavlm_base_plus_cfgv2.0`

- `test/`: 213 generated target WAVs, mono 16 kHz, finite samples, exact per-clip original sample count and verified SHA256 receipts.
- `avhubert_feat/`: 213 finite generated AV-HuBERT features; each shape matches its GT counterpart.
- `_sim_results.jsonl`, `_wer_results.jsonl`, `_emosim_results.jsonl`, `_avsync_results.jsonl`: each has exactly 213 sample records plus a summary. Blank/footer lines are not samples.
- `inference_summary.json`: checkpoint SHA, complete inference protocol, decoder/speaker/normalization metadata and output receipts. Synthesis elapsed time: **469.35 s**.
- `evaluation_audit.json`: recomputed full-precision metrics and complete coverage; `evaluation_per_clip.jsonl` identifies every sample by full relative clip ID.
- `comparison_protocol_with_hard9_150k.json` and `comparison_with_hard9_150k.json`: machine-readable baseline protocol and metric comparisons.

Runtime logs in this project (excluded from Git): `logs/eval_s1_150000_wavlm_tra_20261102.log`, `logs/audit_s1_150000_wavlm_tra_20261102.log`, `logs/evaluation_wavlm_tra_s1_150k_20261102.json`. Detached pipeline PID was **327829** and has exited after completing every stage.

Experiment source-tree hash before the evaluation report was added: `df5ad339a0db945f61dcacf4369b2e8d6c58b649`.

## Reproduction

Run from this snapshot with the existing environment and a new empty output directory. The active shell wrappers already select this experiment's config/checkpoint directory.

```bash
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=150000 EVAL_GPU=0 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  OUTPUT_DIR=/path/to/new_empty_result_directory \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_150000_wavlm_tra_rerun.log 2>&1 < /dev/null &
```

After all stages finish, run `PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python scripts/audit_avt_joint_s1.py --output-dir /path/to/new_empty_result_directory`.
