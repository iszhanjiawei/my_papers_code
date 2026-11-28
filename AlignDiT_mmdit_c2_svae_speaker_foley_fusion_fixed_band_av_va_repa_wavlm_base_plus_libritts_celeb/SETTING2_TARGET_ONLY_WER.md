# Setting2 original 115 targets: target-only text ablation

Use the original 115 CelebV-Dub Setting2 GT-dialogue pairs and the historical
150000-update EMA checkpoint. The only inference intervention is
`--text-input-mode target-only`: `model.sample(text=[target_text], ...)` receives
no reference transcript or separator. The default `reference-target` mode retains
the historical concatenation, including its separator.

The reference waveform still supplies its complete Semantic-VAE latent prompt
and CAM++ embedding. Target video, paired reference, normalized target dialogue,
train-only latent normalization, 1000k Semantic-VAE decoder, seed 0, Euler/EPSS
32 NFE, sway -1, text/video CFG 5/2, FP32 and output cropping remain fixed.
The original manifest and target-text directory are reused. Every output JSON
records the actual `model_text` and its mode. All 115 target-only durations equal
the historical prompt/target/sampler lengths; the verifier enforces this.

Run from this project root, with a free GPU (the launcher creates no session on
its own; use `setsid` for background execution):

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 EVAL_GPU=0 \
  bash src/aligndit/run/eval/eval_setting2_target_only_wer.sh \
  > logs/setting2_target_only_wer_150k.log 2>&1 &
```

Defaults under `${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/Video-to-Speech-benchmark/`:

- Baseline: `results/setting2_gttext_dubbing/Ours_150k`.
- Ablation: `results/setting2_gttext_dubbing/Ours_150k_target_only`.
- Separate reference cache: `cache/ours_svae_campplus_setting2_gttext_target_only`.

`OUTPUT_DIR` and `REFERENCE_CACHE` may override the ablation paths. Generation
checks all cached provenance before resuming. This launcher computes WER only,
using the unchanged `scripts/evaluate_setting2.py` and its local
faster-whisper-large-v3, English, beam size 5, historical punctuation removal and
lowercasing. Corpus WER is total word edits / 1721 reference words, not an average
of sentence percentages. The verifier independently recomputes edit distances,
checks all 115 WAVs, compares every reference latent/speaker hash and video/audio
geometry with the historical run, and writes `target_only_wer_verification.json`.

The historical S1 helper file changed after the original run in commit `a90188a`
to support S1 sharding and memory reporting. Setting2 imports only helper
functions and does not call its S1 `run` or `parse_args`. Excluding those two
functions, its entire AST matches the baseline file (git blob `2cc463b` at commit
`7032540`); this equality is pinned and checked by the verifier. All other model
and sampling source hashes, except the modified dubbing adapter, must equal the
historical run.

Regression checks:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python \
  -m unittest discover -s tests -p 'test_setting2_target_only.py' -v
```

Runtime WAVs, ASR transcripts, logs, caches, and reports are not committed.
