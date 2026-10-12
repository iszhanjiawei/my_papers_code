# GRID validation Setting1 evaluation

This experiment evaluates its GRID-trained checkpoint on all **3,274** audited
validation clips. Training clips are excluded. The reference is the same clip's
complete canonical MPG audio, matching the existing Setting1 oracle prompt
protocol. It is not an independent utterance or speaker reference.

Generation uses initialized EMA, FP32, seed 0 per clip, Euler/EPSS, 32 NFE,
sway −1, text/video CFG 5/2, the fixed LibriSpeech latent normalization and
pretrained vocabulary. Input conditions are the GT Semantic-VAE latent prompt,
40 Hz video-only AV-HuBERT features and same-clip CAM++ vector. The generated
continuation is decoded with the pinned 1000k Semantic-VAE; the prompt is
removed and only the exact target duration is saved as 16 kHz mono PCM16.
One utterance is sampled at a time, preserving the historical sampling protocol;
multiple independent processes per GPU improve throughput without batching.

Four metrics use full relative identities `val/speaker/utterance`, including
clips with repeated basenames across speakers:

- SPKSIM: existing WavLM-Large/ECAPA cosine against canonical same-clip GT.
- WER: local faster-whisper Large-v3, English, beam 5, FP16. Both texts undergo
  NFKC, case folding, Unicode punctuation-to-space conversion, whitespace
  collapse, and isolated digits 0–9 to English words. Alphanumeric boundaries
  are separated (`B2` → `b two`). All GRID letters, including `a`, are retained;
  lexical recognition errors are not corrected. Raw and normalized strings and
  substitution/deletion/insertion counts are retained. Corpus WER is total
  word edit distance divided by total reference words (**19,644**), never a
  mean of sentence WERs.
- EMOSIM: existing emotion2vec classifier-score cosine against canonical GT.
- AVSync: frame-mean cosine between **joint** audio/video 25 Hz, 1024-D
  AV-HuBERT features extracted with the same original grayscale mouth frames
  and generated/GT audio respectively. The training video-only 40 Hz cache
  is not the GT joint-feature reference.

Each output WAV has a receipt binding its hash to checkpoint, configuration,
manifest, normalization and sampling protocol. Resume verifies existing files.
Small `--limit` runs are explicitly marked as diagnostics. Full inference and
all four JSONL result sets must match the validation manifest; merge independently
recomputes corpus WER and verifies coverage. Runtime artifacts stay outside Git.

Run from this project (adjust `ROOT_PREFIX` as in `env.sh`):

```bash
OUT="${ROOT_PREFIX}/zjw524/projects/data/evaluations/GRID_s1_100k_seed0_nfe32_cfg5_2"
mkdir -p "$OUT"
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/eval/eval_grid_s1_100k.sh \
  --output-dir "$OUT" --gpus 0,1 --workers-per-gpu 4 \
  > "$OUT/pipeline.log" 2>&1 &
```

Each worker has its own session and one visible GPU; eight workers provide four
processes per GPU. `pipeline_status.json`, `active_workers.json` and individual
logs record progress. Failed stages preserve verified artifacts. To resume,
repeat the same command; or use `--start-stage sim/wer/emosim/avsync/merge` after
inference has been verified. Keep the same worker count when reusing metric shards.
Local metric weight binaries, software versions and source hashes are recorded in
`metric_assets.json`. Final artifacts are `evaluation_summary.json`, `{sim,wer,emosim,avsync}_results.jsonl`,
`val/*/*.wav`, and joint-feature caches. Metric values are displayed to five decimals.


## Full validation result — 2026-10-12

EMA update **100,000**, checkpoint SHA256
`6394f7e7682f5d096c29a8769d3158f44949746fa84f0c1a66a09ea1dad37da7`.
Two RTX 4090 GPUs each ran four independent processes (eight total), with one
utterance sampled per process. All four metrics cover **3,274/3,274** validation
clips. No additional clips were skipped during inference or evaluation.

| Checkpoint | Coverage | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| GRID EMA 100k | 3274/3274 | 0.81041 | 0.12935 | 0.87838 | 0.78267 |

WER is a fraction: **2,541 word errors / 19,644 reference words**,
equivalent to **12.93525%** after symmetric text normalization. Independent
recomputation found 2,116 substitutions, 334 deletions and 91 insertions. The other
metrics are equally weighted sample means. These values describe the same-clip
Setting1 prompt protocol on GRID, separately from CelebV-Dub results.

Artifacts: `/zjw524/projects/data/evaluations/GRID_s1_100k_seed0_nfe32_cfg5_2`.

- `evaluation_summary.json`: complete four-metric summary and checkpoint identity.
- `verification.json`: independent coverage, finite-value, WAV duration/hash,
  joint-feature shape/hash and corpus WER verification.
- `{sim,wer,emosim,avsync}_results.jsonl`: exactly 3,274 full-ID records each;
  WER retains raw/normalized texts and substitution/deletion/insertion counts.
- `val/*/*.wav`: generated target continuations.
- `avhubert_joint/{gen,gt}/val/*/*.npy`: two sets of 3,274 joint features.
- `protocol.json`, `metric_assets.json`, receipts and worker logs: provenance.

Runtime artifacts remain outside Git. All evaluation workers have finished.
