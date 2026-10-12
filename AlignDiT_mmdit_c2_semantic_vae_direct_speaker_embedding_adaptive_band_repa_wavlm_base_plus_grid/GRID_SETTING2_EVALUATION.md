# GRID Setting2: real-dialogue dubbing, EMA 100k

The target text is **always the real GRID dialogue**. No lip-reading transcript
is used. This is Setting2 GTText dubbing, on the same 3,274 audited validation
targets as Setting1, with a different utterance from the **same speaker** as
reference. For each target, its own GT audio, latent and CAM++ vector are not
used by generation. Target video-only AV-HuBERT features and real target text
remain inputs; the independent reference provides both latent prompt and CAM++.

Reference assignments use only the complete validation partition. Within each
speaker, sort IDs by SHA256(`0:val/speaker/utterance`), then take the first cyclic
shift that makes every reference different in ID **and normalized transcript**.
This provides a deterministic seed-0 permutation: each validation clip is used
once as a reference for another clip. Freeze the 3,274 assignments and canonical
audio/latent/speaker hashes in `reference_pairs.json`; no selection depends on
model outputs or metric scores. Smoke runs use the same full assignment with
only their requested targets generated.

Reuse the verified reference caches: 64-D/40-Hz Semantic-VAE posterior samples,
CAM++ 192-D vectors, fixed LibriSpeech normalization and original vocabulary.
Prompt and target lengths can differ. Concatenate reference real dialogue,
two spaces, target real dialogue. Prompt video is zero, target video starts at
the independent prompt's end. Decode only target frames and crop to the target
video duration (`25 Hz frame count × 640 samples`) as 16-kHz mono PCM16.

Keep Setting1's EMA 100k, FP32, seed 0 per target, Euler/EPSS, 32 NFE, sway −1,
text/video CFG 5/2, and one utterance per process. Run four processes per GPU
on two RTX 4090s, eight independent workers total. If other workloads share the
GPUs, Whisper shards wait for at least 8,192 MiB of free memory and serialize
model allocation. Two model slots per GPU remain held until each ASR shard
finishes; all eight shards are retained without changing ASR parameters.

Metrics retain the existing implementations and Setting1 text normalization:

- SPKSIM compares generated speech with the **independent reference actually
  used by generation**, using WavLM-Large/ECAPA. Its reference differs from
  Setting1's same-clip GT, so the SPKSIM difference reflects both this metric
  pairing and the generation protocol.
- WER uses real target dialogue and local faster-whisper Large-v3 ASR; normalize
  both texts (NFKC, case, punctuation, whitespace, digits), preserve GRID letters
  including `a`, and aggregate word errors over all **19,644** reference words.
- EMOSIM compares emotion2vec classifier-score vectors with target canonical GT.
- AVSync compares joint 25-Hz AV-HuBERT features from the same target mouth video
  with generated and target GT audio. The video-only training cache is not the
  joint-feature reference.

All four result sets must cover every target ID; WAV receipts bind output hashes
to checkpoint, pair assignment, configuration and sampling. Partial smoke results
are explicitly marked and cannot be reported as full validation results.

```bash
OUT="${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_s2_gttext_100k_seed0_nfe32_cfg5_2"
mkdir -p "$OUT"
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/eval_grid_s2_100k.sh \
  --output-dir "$OUT" --gpus 0,1 --workers-per-gpu 4 \
  > "$OUT/pipeline.log" 2>&1 &
```

`pipeline_status.json`, worker logs and `active_workers.json` record progress.
Reuse the same output, pairing and worker count to resume verified artifacts;
`--start-stage sim/wer/emosim/avsync/merge` starts after verified generation.
The final `evaluation_summary.json`, four full-ID JSONL files, generated WAVs,
reference assignments and joint-feature caches stay outside Git.

## Full validation results, 2026-10-12

All generation and metric stages completed on **3,274/3,274** targets. The
checkpoint is EMA `model_100000.pt`, SHA256
`6394f7e7682f5d096c29a8769d3158f44949746fa84f0c1a66a09ea1dad37da7`.
Reference assignments have SHA256
`eb092eeda9294fa06a21d893c1a04ce09e20bf3d2bb0fdd99558ad0e5acbd99c`.

| Setting | Coverage | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| Setting1, same-clip prompt | 3274/3274 | 0.81041 | 0.12935 | 0.87838 | 0.78267 |
| Setting2, independent same-speaker prompt, GTText | 3274/3274 | 0.66391 | 0.47180 | 0.81068 | 0.70023 |

Setting2 corpus WER is **9268 / 19644 = 0.4717980045**, or **47.17980%**:
S=7820, D=819, I=629. All target and ASR texts use the normalization above;
no words are corrected, no utterances are dropped, and no per-shard WER means
are averaged. Setting1 WER is 2541 / 19644 (12.93525%). These are descriptive
results under different prompt protocols. SPKSIM also changes its metric
reference from same-clip GT to the independent prompt; its difference cannot
be attributed solely to generation. EMOSIM and AVSync retain target GT as
their comparison reference in both settings.

Independent verification passed with **3,274 successes and zero failures**.
It rechecked all WAV formats, durations and hashes, reference permutations and
cached reference hashes, prompt/target geometry, four full-ID result sets and
their aggregates. WER was recomputed with separate dynamic programming and
jiwer. All 6,548 joint features passed shape, FP32, finite-value, hash and
audio/mouth/model binding checks; independent NumPy AVSync recomputation had
maximum absolute difference `1.320251749e-7`. Eleven metric assets and the
EMA checkpoint were rehashed. The evaluation processes have finished.

Artifacts:

- [Full four-metric summary](/zjw524/projects/data/evaluations/GRID_s2_gttext_100k_seed0_nfe32_cfg5_2/evaluation_summary.json).
- [Independent verification](/zjw524/projects/data/evaluations/GRID_s2_gttext_100k_seed0_nfe32_cfg5_2/verification.json).
- `sim_results.jsonl`, `wer_results.jsonl`, `emosim_results.jsonl` and
  `avsync_results.jsonl` each contain 3,274 full IDs in validation-list order.
- `reference_pairs.json`, `protocol.json`, `metric_assets.json`, WAV receipts,
  `avhubert_joint/{gen,gt}/` and logs bind pairs, configuration and model assets.
  `metric_assets.json` preserves the initial source revision and the subsequent
  WER resume revision `3f18e82`, which changes GPU allocation only.
