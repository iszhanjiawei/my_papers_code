# CelebV-Dub targets with independent GRID references

This protocol evaluates the existing REPA lambda=0.1 EMA checkpoint at update
150000, without fine-tuning. It uses exactly 213 target clips and one GRID
reference utterance per target, not a target-by-speaker Cartesian product.

## Fixed inputs

- Targets: the historical CelebV-Dub Setting 1 list, in its original order.
- References: the local GRID `audio_25k` waveforms with matching `.lab`
  transcripts in `Grid_resample_ABS/Grid_Wav_22050_Abs`. The 33 speakers with
  transcripts are balanced across targets; s21 is excluded because this local
  copy lacks its transcripts. Pairing seed is 0. Silence annotation tokens
  (`sp`, `sil`, `<sil>`) are removed from labels, preserving the six spoken words.
- Use complete reference utterances. No extra silence trimming, stretching,
  target-dependent content selection, or concatenation. Local GRID WAV durations
  differ from the original alignment timestamps; do not assume a 3-second prompt.
- The reference is resampled to 16 kHz mono and supplies both the Semantic-VAE
  latent prompt and the frozen CAM++ embedding. The existing 1000k EMA VAE,
  posterior-sample protocol (base seed 666), train-only normalization statistics,
  and CAM++ extraction contract are retained. Legacy GRID features are not used.
- Text is GRID transcript followed by the historical separator and target text.
  Both the reference and target occupy the same 40-Hz timeline. Video consists
  of a zero placeholder for reference frames followed by target video features.
- Duration is known from the historical target metadata. No target waveform or
  target acoustic latent is loaded by generation. Only the target output segment
  is decoded and saved, cropped to the target's original sample count.
- Inference: EMA, seed 0, Euler/EPSS, 32 NFE, sway -1, text/video CFG 5/2.
  Text-induced duration extension is explicitly padded and recorded; inputs
  exceeding the sampler's 4096-frame limit fail during preflight.

The manifest preserves full target, reference, and pair identifiers, paths,
transcripts, lengths, and artifact hashes. All compared checkpoints must use the
same manifest. GRID is absent from this run's CelebV-Dub training manifest and
the direct LibriSpeech initialization stage; this does not establish that every
external pretrained component has never seen a GRID speaker.

## Entry point

From this project root, with the existing aligndit environment:

```bash
setsid env PYTHONUNBUFFERED=1 EVAL_GPU=0 \
  bash src/aligndit/run/eval/eval_celebvdub_grid_reference_150k.sh \
  > logs/eval_grid_reference_150k.log 2>&1 &
```

`RUN_STAGE=prepare`, `infer`, or `metrics` runs an individual stage. The default
`all` runs preparation, generation, metrics, and independent verification. The
preparation tool validates reusable caches. Generation refuses a nonempty output
directory. The metrics stage skips completed summary files, then independently
verifies all outputs; it can resume after a failed metric stage.

`GRID_CACHE_DIR` and `OUTPUT_DIR` override runtime locations. Defaults are under
`/zjw524/projects/data/`, prefixed by `ROOT_PREFIX` where applicable. The default
reference directory is `Grid_reference_celebvdub_s1_seed0_svae1000k_campplus_v1`;
the output directory under the existing checkpoint folder is
`eval_gridref_150000_repa_baseplus_pairseed0_cfgv2.0`.

## Metric semantics

| Task | Comparator | Aggregation |
|---|---|---|
| `sim` | Actual GRID reference waveform; WavLM speaker similarity | Sample mean |
| `wer` | CelebV-Dub target transcript; existing ASR and normalization | Total word edits / total reference words |
| `emosim` | CelebV-Dub target GT; emotion2vec classifier-score cosine | Sample mean |
| `emoembed` | CelebV-Dub target GT; emotion2vec utterance-embedding cosine | Sample mean |
| `avsync` | AV-HuBERT joint-feature similarity between target video with generated audio and target video with GT audio | Frame mean, then sample mean |

The historical AVSync score is not a speaker-invariant, independent lip-sync
measurement. Emotion metrics and AVSync are auxiliary diagnostics in this
cross-speaker protocol. No UTMOS or SyncNet score is claimed by these scripts.
Historical same-clip-reference results must be reported as a separate protocol.

Every metric JSONL contains 213 unique full target/pair IDs. The final verifier
checks WAVs, reference caches, AV features, manifest bindings, and independently
recomputed metrics before writing `_verified_summary.json`. Runtime audio,
caches, checkpoints, logs, and metric outputs are not committed to Git.

## CPU regression checks

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python \
  -m unittest discover -s tests -p 'test_grid_reference_*.py' -v
```

These checks cover independent reference/target lengths, real tiny-model CFG
sampling, text-induced extension, the sampler duration limit, deterministic
pairing, separate speaker/emotion comparators, full identifiers, corpus WER,
and manifest/output integrity.

## Verified run: 2026-09-27

GPU 0 (RTX 4090) completed all 213 pairs using EMA update 150000. The 28 CPU
tests, three-item real-weight smoke run, and full independent verifier passed.
The first three formal WAV hashes equal the smoke outputs. No pair required a
text-duration extension. All 213 WAVs and AV features were verified; total target
audio duration is 721.059 seconds.

| SPKSIM | Corpus WER | EMOSIM (scores) | AV-HuBERT similarity | EMO embedding cosine |
|---:|---:|---:|---:|---:|
| 0.47041 | 0.05593 | 0.58239 | 0.51039 | 0.94584 |

Corpus WER is 133 / 2378 = 5.59294%. The pairing manifest SHA256 is
`cd1ab0adf681946c389a5e674eb44d0c923a686a0f9bcf5418794d0a3e72c729`.
The default output directory contains `inference_summary.json`, five metric
JSONL/summary pairs, and `_verified_summary.json`. Full settings, evidence, and
limitations are recorded in section 27 of
[`实验结果总汇.md`](../实验结果/实验结果总汇.md).
