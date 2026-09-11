# Isolated visual feature path-distance alignment

This is a real source copy of the fixed-band speaker C2 snapshot (`6c69daa`),
recorded before modification by commit `c45029a`. Baseline, adaptive-band and
fixed-band siblings are untouched. Runtime logs, events, outputs and checkpoints
were excluded from the copy. No shared source links or cache rewrites are used.

## Exact intervention

For frozen native 25-Hz AV-HuBERT video features, compute in FP32:

```
u[r] = video[r] / max(norm(video[r]), 1e-12)
c[0] = 0
c[r] = c[r-1] + norm(u[r] - u[r-1])
B[i,j] = -0.5 * ((j/40 - i/40) / 0.100)**2
         -0.5 * ((c40[j] - c40[i]) / 2.0)**2
```

Only audio-query/video-key logits in the first 12 joint-attention blocks receive
this additive prior. The AA, VA and VV score rules, text cross-attention, final
six speaker-conditioned blocks, flow/CTC losses and optimizer protocol are
unchanged. The shared joint softmax is retained. There is **no** predictor,
trainable parameter, auxiliary encoder, teacher, new loss, center shift or
attention-mass correction. The nonpositive bias can reduce video attention mass
at fixed content logits, so quality differences must not be attributed solely
to temporal routing without further controls.

The scalar path is accumulated **before** interpolation. `c40` uses the same
linear `align_corners=False`, exact-output-length coordinate convention as the
existing cached 25-to-40-Hz video features. This includes the cache's half-pixel
sampling/edge-clamping convention: it is not a separately resampled `j/25` grid.
Existing 40-Hz video arrays themselves are never recomputed or modified.
There is no per-clip path-length normalization or derivative estimate from
already-interpolated/trainable embeddings. All model-time bias distances stay
FP32 before the existing SDPA mask cast.

The path sigma **2.0** is dimensionless, and time sigma **100 ms** is a Gaussian
standard deviation, not a hard cutoff. This initial engineering scale was chosen
after inspecting 100 training-only clips (manifest indices 0, 500, ..., 49500):
native increment median 0.71777; cumulative distance across three native intervals
(120 ms) median 2.17381, 10th/90th percentiles 1.16819/2.84751. This is not a
quality-metric search or evidence of improved synchronization. Different path
scales require separate runs and matching contract sidecars.

The model requires explicit `video_path` in visual-path mode; missing native
information is an error, not a silent 40-Hz approximation. Invalid padding is
end-repeated by the data loader. For conditioning dropout, increments touching
either a complementary-hidden or padded endpoint are removed and the remaining
increments are reaccumulated. Null-video CFG branches use a constant path, so
original path content cannot enter the unconditional branch. This removes
masked edges, not the pre-existing contextual receptive field of AV-HuBERT.
The same policy is used for training, separate/packed CFG and cached inference.
In Setting 1, the dummy-video prompt gets a constant path followed by a rebased
target path; there is no fictitious prompt-to-target feature jump.

## Matched training protocol

- SHA-pinned S2c **70k EMA** parent; fresh optimizer/update counter.
- Seed 666; rank RNG 666 + rank; 4 RTX 4090; BF16; 3,600 latent frames/GPU.
- LR 5e-5; 20k LR warmup; inherited 200-epoch decay horizon; stop at 200k updates.
- CTC zero through 10k, linear to 0.03 at 30k, unchanged taps at blocks 6/12.
- All 79,613 training examples retained, including 105 CTC-infeasible examples
  handled by the inherited `zero_infinity` policy.
- Numbered checkpoints every 50k, `model_last.pt` every 5k.
- Strict parent migration retains 313 source / 704 target / 303 loaded /
  10 ignored / 401 new state tensors; path alignment adds no serialized tensors.

Dedicated config:
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_visual_path_band.yaml`.

Checkpoint directory under `ROOT_PREFIX`:
`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_visual_path_band_ctc003_warmup10k30k_40hz_CelebVDub_char`.

```
bash scripts/start_visual_path_band_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_visual_path_band_4x4090.sh \
  > logs/train_visual_path_band.log 2>&1 < /dev/null &
```

The launcher refuses occupied GPUs. Smoke runs must use distinct checkpoint and
model names. Never resume a trained baseline/fixed/adaptive checkpoint as if it
were this new experiment. Copied generic/historical launchers remain only for
regression compatibility; use the dedicated visual-path entry here.

TensorBoard serves this snapshot's `runs/`, default port **6010**. Formal run:
`AlignDiT_MMDiT_c2_svae_speaker_visual_path_band_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`.
Only rank 0 writes total, flow, CTC/weighted CTC losses, LR, global/speaker
gradient diagnostics and temporal/path diagnostics. Local-batch scalars are not
all-GPU averages; there is no predictor-gradient scalar for a parameter-free rule.

## Inference and regression checks

The copied speaker inference/evaluation shell entries now default to the new
config, checkpoint and output directory. The existing Setting 1, same-clip
reference, EMA, seed 0, text/video CFG 5/2 and 32 NFE protocol is preserved.
Keep `speaker_training_contract.json` beside moved checkpoints: strict tensor
loading alone cannot distinguish these parameter-free experiments. Inference
rejects missing, wrong-mode or mismatched path/time settings.

```
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_visual_path_temporal_band.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_fixed_temporal_band.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_adaptive_temporal_band.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
```

This implements a candidate temporal prior, not a verified innovation or quality
gain. Frozen visual representation changes can reflect nuisances as well as
speech-related changes. The path prior can also suppress useful anticipatory
visual evidence; it is not an unknown-lag estimator. Later tests should compare
real paths against shuffled-increment paths and bandwidth/mass-matched temporal
controls, using generated-speech AVSync and content accuracy, not attention
heatmaps alone.

### Pre-launch checks (2026-09-12)

- 31 visual-path CPU tests passed, including native accumulation/interpolation,
  loader failures, collate padding, same-clip prompt assembly, path-source
  contracts, CFG/cache/mask isolation, AV-only reference attention, exact legacy
  disable behavior, unchanged RNG/state, EMA and checkpointed gradients.
- The inherited 25 fixed-band tests, 19 adaptive-band tests and speaker suite
  passed unchanged. New/targeted files pass Ruff; inherited core files also pass
  focused error checks with pre-existing annotation/long-line exceptions.
- All 79,613 original native train feature files passed read-only availability,
  path-confinement, shape and float32-header validation (73.37 s, eight workers).
  This was not a full-array finiteness audit; finite contents remain checked on
  every real sample load. No native or 40-Hz cache file was modified.
- Full-batch four-GPU BF16 training completed three smoke updates and wrote
  `logs/ddp_visual_path_band_smoke/model_last.pt` in a separate smoke directory.
  Strict parent migration retained the expected 313/704/303/10/401 counts.
- That checkpoint passed strict EMA/online reload and real-data backward tests:
  total loss 1.04999 at CTC 0; total loss 1.36486 at CTC 0.03; finite nonzero
  existing-model gradients in both. The native-derived path changes the model's
  output relative to a zero/static path at matched RNG. Detailed functional
  results are in ignored `logs/visual_path_band_checkpoint_validation.json`.
- One real Setting 1 test example completed EMA -> latent -> Semantic-VAE
  decoder -> valid WAV using two NFEs. This was a functional smoke check only,
  not a 32-NFE benchmark or a synchronization evaluation. Its metadata is in
  `logs/visual_path_band_inference_smoke/inference_summary.json`.

These are functionality checks, not speech-quality or synchronization results.
