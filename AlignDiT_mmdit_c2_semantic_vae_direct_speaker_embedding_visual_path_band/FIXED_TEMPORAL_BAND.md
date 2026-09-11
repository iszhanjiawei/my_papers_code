# Isolated Semantic-VAE C2 + speaker + fixed temporal band

This snapshot is a real source copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band` at
`098b580`, which was itself copied from the speaker baseline. Neither sibling
is modified. Logs, events, checkpoints, outputs and caches were not copied.
The pinned existing data, frozen codec and speaker caches are read-only inputs.

## Intervention

Only audio-query/video-key logits in the first 12 joint blocks receive:

`B[i,j] = -0.5 * ((j/40 - i/40 - 0.0) / 0.100)**2`.

- Center is the audio query's physical time: offset **0 seconds** throughout.
- Width is **sigma 100 ms**, shared across examples, heads, layers and updates.
  Sigma is a Gaussian standard deviation, not a hard cutoff or a 100-ms total window.
- This matches the adaptive band's initial mathematical prior, but has **zero
  trainable parameters**: no predictor, frozen MLP, trainable buffers or new loss.
- FP32 distances use the model's already-interpolated 40-Hz audio/video grids.
  No cache interpolation, duration convention, conditioning dropout or speaker
  injection changes. The prior also remains active for dropped-video CFG branches.
- AA, VA and VV logits are untouched. The existing joint softmax is retained;
  a nonpositive AV prior also reduces total video attention mass at fixed logits.
  No hard mask, mass correction, teacher, auxiliary encoder or progressive window.
- Active in both training (including no padding-attention-mask mode) and inference.
  Disabling the feature recovers the existing legacy forward path and state keys.

This is the fixed-band control for the adaptive-band experiment, not a new
attention principle or a claim of measured quality improvement.

## Matched protocol and isolated launch

Initialize from the identical SHA-pinned S2c **70k EMA** parent with a new optimizer
and update counter. Do not resume a speaker/adaptive trained checkpoint. Retain
seed 666 (rank RNG 666 + rank), 4 GPUs, bf16, 3,600 latent frames/GPU, LR 5e-5,
20k LR warmup, original 200-epoch decay horizon, CTC zero through 10k and linear
to 0.03 at 30k, and stop at 200k updates. Retain all 79,613 training records,
including 105 CTC-infeasible records handled by the existing zero-infinity policy.
Numbered checkpoints every 50k; `model_last.pt` every 5k. Strict parent migration
expects 313 source / 704 target / 303 loaded / 10 ignored / 401 new tensors,
exactly the speaker baseline's tensor schema (no temporal-band tensors).

```bash
bash scripts/start_fixed_band_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_fixed_band_4x4090.sh \
  > logs/train_fixed_band.log 2>&1 < /dev/null &
```

Use only the fixed-band launcher/config for this experiment. Copied adaptive and
older generic configurations remain for regression compatibility; they describe
other experiments and must not be used to launch into their existing run paths.

Config: `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_fixed_band.yaml`.
Checkpoint directory under `ROOT_PREFIX`:

`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_fixed_band_ctc003_warmup10k30k_40hz_CelebVDub_char`

TensorBoard serves this snapshot's `runs/`, port **6009** by default. Formal run:

`AlignDiT_MMDiT_c2_svae_speaker_fixed_band_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`

Only rank 0 writes scalars: total/flow/CTC losses, LR, existing global/speaker
gradient diagnostics, and fixed offset/sigma mean/std/min/max in milliseconds.
These are local-batch diagnostics, not all-GPU averages. There is deliberately no
predictor gradient scalar. The training contract records fixed mode and all
physical-time constants. Smoke runs use separate names/directories.

## Verification and inference

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_fixed_temporal_band.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_adaptive_temporal_band.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
```

Use `validate_fixed_band_checkpoint.py` after a short real-data run to validate
strict EMA/online loading, fixed constants and finite real-data gradients with
CTC weights 0 and 0.03. Validation performs no optimizer update.

The copied `infer_celebvdub_s1_svae_direct_speaker_ctc003.sh` and corresponding
`eval_...sh` now select the fixed-band config/checkpoint/output directory. Existing
Setting 1 same-clip reference, EMA, seed 0, CFG text/video 5/2 and 32 NFE are retained.
Because fixed priors have no state keys, inference additionally requires the
matching `speaker_training_contract.json` beside the checkpoint; keep this file
when moving weights. A baseline/fixed or mismatched-width config is rejected
instead of silently loading an indistinguishable tensor schema.

`ADAPTIVE_TEMPORAL_BAND.md` is inherited historical documentation for the source
experiment, not instructions to run this fixed-band control. Runtime validation
reports, training logs, events and generated smoke audio remain ignored by Git.

### Pre-launch verification (2026-09-12)

- All 25 fixed-band CPU regressions passed, including config/sidecar rejection,
  exact inherited-state/RNG preservation, manual joint-softmax reference,
  initial adaptive-band parity, CFG, padding masks and checkpointed gradients.
- The copied 19-test adaptive suite and existing speaker suite passed unchanged.
- Four-GPU BF16 real-data smoke completed three updates at the full
  3,600-frame/GPU batch; expected strict S2c migration counts all matched.
- The resulting EMA/online checkpoint reloaded strictly. Real two-example
  backward tests passed with CTC 0 and 0.03, finite losses/global gradients and
  unchanged fixed constants. No band parameters/state were present.
- One real Setting 1 sample completed the full EMA -> latent -> VAE -> WAV path
  using two NFEs, solely as a functional smoke test.

These checks establish functionality, not AVSync or speech-quality improvement.
