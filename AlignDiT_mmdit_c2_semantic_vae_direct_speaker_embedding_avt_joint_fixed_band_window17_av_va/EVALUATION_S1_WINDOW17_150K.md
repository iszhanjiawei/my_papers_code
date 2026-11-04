# Window17 Gaussian AV/VA band: 150k EMA evaluation

Evaluation and full coverage audit completed on 2026-11-04. This report belongs
to the window17 experiment: fixed sigma 0.200 seconds, nominal +/-8 frames at
40 Hz (17-frame soft core), with Gaussian tails retained.

## Training stopped and checkpoint selected

The numbered `model_150000.pt` checkpoint was present and its archive was valid.
At the user's request, the window17 training launcher and all its worker processes
were stopped; the last update observed when requesting the stop was 151059.
Evaluation uses the saved **150000 EMA** weights. Training remains stopped.

## Results and matched historical comparisons

All rows use 150k EMA and the same 213-clip CelebV-Dub Setting 1 protocol.
The 5/9/13-frame rows are read from their existing audited outputs and were not
rerun in this task. Generation settings, decoder metadata, speaker metadata,
normalization, latent-cache root and test-list hash match the current run.

| Gaussian soft core | Sigma | SPKSIM ↑ | WER ↓ | EMOSIM ↑ | AVSync ↑ |
|---|---:|---:|---:|---:|---:|
| 5 frames | 50 ms | 0.64788 | 0.04794 | 0.77171 | 0.56360 |
| 9 frames | 100 ms | 0.64895 | 0.04668 | 0.74785 | 0.57954 |
| 13 frames | 150 ms | 0.65271 | 0.05088 | 0.75905 | 0.58568 |
| **17 frames (current)** | 200 ms | 0.65139 | 0.04373 | 0.77312 | 0.57128 |

Current corpus WER is **104 word errors / 2378 reference words = 4.37342%**.
Compared with 13 frames, this run has lower WER and higher EMOSIM, with lower
SPKSIM and AVSync. These are descriptive comparisons of independently trained
runs on the same test set.

## Protocol and checkpoint identity

- All 213 clips, same-clip full GT reference prompt and CAM++ speaker vector,
  real target duration; generated WAVs contain only the target segment.
- EMA, seed 0, CFG text/video 5/2, Euler with EPSS, sway -1 and 32 NFE.
- The matching window17 YAML and fixed-band training contract explicitly
  select sigma 0.200 seconds. Model, training and evaluation code are unchanged
  by this evaluation task.

Checkpoint root:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window17_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Checkpoint: `model_150000.pt` (update 150000, EMA).
SHA256: `3a19b12a046bb8b78397c92294f276de73ec6f074327dfbdb95d34a320427e83`.

Output subdirectory:
`eval_s1_150000_speaker_avt_joint_fixed_band_window17_av_va_cfgv2.0`.

## Verification and artifacts

The completed audit checks exactly 213 WAVs, 213 generated AV-HuBERT feature
arrays and 213 records per metric. It preserves full video/clip identities,
including repeated clip basenames in different videos. WAVs are finite, nonempty,
16-kHz mono, have receipt-matched sample counts and SHA256. All generated/GT
features are finite and their shapes agree. AVSync is independently recomputed
from features; sample means and corpus WER are recomputed from result records.
Summary lines and blank lines are excluded from sample counts.

The output contains `inference_summary.json`, `test/`, `avhubert_feat/test/`,
the four `_*_results.jsonl` files, `evaluation_per_clip.jsonl` and
`evaluation_audit.json`. Logs, audio, features and checkpoints are not committed.

## Reproduction and run records

The existing evaluation entry was run from this project root with:

```bash
source env.sh
setsid env PYTHONUNBUFFERED=1 CKPT_STEP=150000 EVAL_GPU=0 \
  CFG_TEXT=5.0 CFG_VIDEO=2.0 NFE=32 INFER_SEED=0 \
  bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh \
  > logs/eval_s1_150000_window17_cfgv2.0_20261104.log 2>&1 < /dev/null &
```

For a new run, set `OUTPUT_DIR` to a fresh directory. The entry refuses to
overwrite existing generated audio. Audit a completed output with:

```bash
output_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window17_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char/eval_s1_150000_speaker_avt_joint_fixed_band_window17_av_va_cfgv2.0"
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/audit_avt_joint_s1.py --output-dir "$output_dir"
```

Evaluation PID 157819 and audit PID 179620 both completed. Runtime records:

- `logs/training_stop_for_150k_eval.json`
- `logs/evaluation_window17_150k_20261104.json`
- `logs/eval_s1_150000_window17_cfgv2.0_20261104.log`
- `logs/audit_s1_150000_window17_20261104.log`
- `logs/window17_150k_protocol_comparison.json`
- `logs/window17_150k_comparison.json`

Only this results/reproducibility document is added to Git for this evaluation.
