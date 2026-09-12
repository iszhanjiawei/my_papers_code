# LocAt-inspired temporal Gaussian attention

## Isolation and provenance

This directory is a real source copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding`, not a source symlink
and not an extension of the fixed/adaptive/visual-path-band branches.
The unmodified copy is recorded in Git commit `88a4b31`.
Data, frozen feature caches and the pretrained S2c checkpoint remain shared,
read-only inputs. New checkpoints, TensorBoard events and inference outputs use
separate paths. Always launch from this directory with `PYTHONPATH=src`; do not
install this snapshot over another experiment's editable Python package.

Inspiration: **Locality-Attending Vision Transformer**, ICLR 2026,
<https://arxiv.org/abs/2603.04892>, official implementation
<https://github.com/sinahmr/LocAtViT> (`f99aac4`). This is a task-specific 1-D
adaptation of GAug, not a full reproduction of LocAtViT, and does not add PRR.
No improved alignment, perceptual quality or novelty claim is implied by the
implementation or its smoke tests.

## Mechanism

Rows denote queries, columns denote keys. For an audio query at physical time
`tau_a[i]` and video key at `tau_v[j]`, add to the **joint A/V logits**:

```text
B_AV[b,h,i,j] = alpha[b,h,i]
                 * exp(-0.5 * ((tau_v[j] - tau_a[i]) / sigma[b,h,i])**2)
```

`B_AV` is nonnegative: this is a positive Gaussian bump, NOT the negative
quadratic/log-Gaussian used in the previous temporal-band experiments. There
is no Gaussian density normalization, no row normalization before softmax,
no center offset and no hard temporal cutoff. AA/VV logits are unchanged.
Audio and video keys still share the same softmax; text cross-attention is
separate and unchanged.

Each active layer/direction owns `Linear(head_dim, 1)` predictors `log_sigma`
and `log_alpha`. Their weights are shared across heads within that layer, but
outputs depend on each head's normalized query **before RoPE**. AV uses audio
queries; VA, when enabled, uses video queries. Biases are not transposed or
shared between directions. The implementation predicts a bounded **standard
deviation in seconds**, whereas the original image implementation predicts
variance in grid units. The explicit seconds parameterization avoids confusing
variance with standard deviation or copying image resolution heuristics.

Initial transfer settings:

| Setting | Value |
| --- | --- |
| Audio/video time grids | 40 Hz / 40 Hz |
| Sigma bounds | 25–400 ms |
| Initial sigma | 100 ms |
| Alpha | positive unbounded softplus |
| Initial alpha | 0.1 |
| Initial predictor weights | zero, producing a constant initial prior |
| Primary experiment | AV in the first 12 MM layers |
| Optional VA | first 11 MM layers |

The small initial strength and physical bounds are transfer choices, not the
original paper's defaults and not values tuned on test outcomes. The initial
boost is positive, so the enabled model is not exactly the unmodified baseline.
The disabled model retains the original computation and initialization. New
predictors are created without shifting the random initialization of existing
weights.

The current dataset already contains 25→40 Hz resampled AV-HuBERT features.
No new feature extraction, visual encoder, alignment teacher or auxiliary loss
is introduced. Existing CFM supervision is on generated audio frames; it does
not have the original paper's CLS-only supervision bottleneck.

VA updates video for the *next* layer to consume. The last MM video output is
not consumed by the audio-only tail, so no final-layer VA predictor is created.
At 64 dimensions per head, AV adds `12 * 2 * 65 = 1560` parameters, VA adds
`11 * 2 * 65 = 1430`, and AV+VA adds 2990. Head-dependent bias tensors can still
increase runtime and memory despite the small parameter count.

## Conditioning and padding

The prior is disabled on fully dropped video branches (including packed TTS
and unconditional CFG branches). Complementary-hidden or padded video positions
receive no enhancement: AV keys and VA queries are gated accordingly. Audio
padding likewise receives no enhancement. This does not remove the baseline
null-token attention or change its existing attention padding protocol.
In particular, the inherited training loop deliberately passes no block
padding attention mask. That behavior remains unchanged for a controlled
comparison; the enhancement itself still respects valid lengths.

Packed CFG keeps branch-major ordering for batches larger than one. New masks
contain visibility information only, not hidden video features. No raw visual
condition is added to an unconditional branch.

## Configurations and limitations

All configurations are in `src/aligndit/config/`:

| Suffix after `finetune_celebvdub_mm_c2_svae_speaker_locat_` | Mode |
| --- | --- |
| `av.yaml` | AV Gaussian; first formal run |
| `va.yaml` | VA Gaussian only |
| `av_va.yaml` | Independent AV and VA Gaussian predictors |
| `av_uniform.yaml` | Query-dependent uniform video boost, no temporal shape |

The original speaker directory remains the no-locality baseline. The optional
configurations are provided for later controlled runs; creating them does not
automatically launch additional training jobs.

A positive AV-only bias can increase **total video attention mass**, not just
redistribute attention within video. As sigma becomes very large, the bias
approaches a uniform video boost, not the baseline. Alpha approaching zero
closes the enhancement. The uniform configuration is a useful modality-gain
control, but is not automatically matched to the Gaussian model's total video
attention mass. Quality evaluation must distinguish modality weighting from
temporal localization.

Audio query reliability changes with diffusion time. A fixed synchronized
center does not estimate AV lag or prove robustness to anticipatory mouth
motion. Neither a narrow learned sigma nor a lower training loss proves better
synchronization. Compare generated-speech AVSync, WER, EMOSIM and SPKSIM using
the same checkpoint, sampling and reference protocol.

## Training and artifacts

The inherited protocol is unchanged: S2c 70k **EMA** initialization, new
optimizer/update counter, seed 666, four GPUs, bf16, 3600 latent frames/GPU,
maximum 32 samples/GPU, 16 workers/GPU, LR `5e-5`, 20k LR warmup, the original
200-epoch LR horizon, stop at 200k updates. CTC is zero through 10k and warms
linearly to 0.03 at 30k. `model_last.pt` is saved every 5k updates and numbered
checkpoints every 50k. The smoke run uses separate artifact paths and is not
used to initialize the formal run.

Primary checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_locat_av_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Primary TensorBoard run name:

```text
AlignDiT_MMDiT_c2_svae_speaker_locat_av_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

TensorBoard logdir is this project's `runs/`; the default port is 6012. Runtime
PIDs, the verified current-server address, launch logs and validation reports
are recorded under the ignored `logs/` directory, not hardcoded as permanent
machine-specific configuration.

Sigma/alpha diagnostics describe active positions on rank 0, not a global DDP
average. They are zero when rank 0 has no visible-video query in that batch;
check `locat/av/valid_query_fraction` before interpreting a zero as a learned
scale. Predictor gradient norms are measured after DDP reduction and can remain
nonzero because other ranks saw video.

Launch from this project (after validation and an idle-GPU check):

```bash
bash scripts/start_locat_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_locat_4x4090.sh \
  > logs/train_locat_av_4x4090.log 2>&1 < /dev/null &
```

Use `TRAIN_CONFIG` to select an ablation; each shipped configuration has a
different model name and checkpoint path. Checkpoint sidecars record LocAt
semantics and guard resume/inference against incompatible direction, width,
strength, time grid or mode choices. Strict parent migration continues checking
the original 313 source keys, 303 loaded keys and 10 ignored keys; only the
exact configured predictor keys are permitted as additional new state.

The inherited speaker-named inference/evaluation shell entries in this copy
now default to the new AV checkpoint/config. For a different mode, pass both
`CHECKPOINT_DIR` and `INFER_CONFIG`. Historical unrelated launch entries remain
for provenance; use the dedicated LocAt launcher for this experiment.

## Validation entry points

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python \
  src/aligndit/script/misc/test_locat_temporal.py

# After a separately named, short four-GPU smoke run:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python \
  src/aligndit/script/misc/validate_locat_checkpoint.py \
  --checkpoint logs/ddp_locat_smoke/model_last.pt --step 3 \
  --output-json logs/locat_checkpoint_validation.json
```

Validation includes the positive-kernel equation, gradients, directional
quadrants, disabled baseline equivalence, physical grids, CFG/visibility,
checkpoint round trips and strict semantic contracts. GPU validation checks
bf16 distributed updates plus strict EMA/online reload and real-data CFM/CTC
backward passes. A tiny decoded inference smoke test is a functionality check,
not a perceptual-quality evaluation.

### Verified before the first formal launch (2026-09-12)

- 31 CPU mechanism/integration regressions and 22 contract/migration tests pass;
  the inherited speaker smoke suite also passes.
- Four-GPU bf16 smoke training completes three updates with the unchanged
  3600-frame budget, saving online/EMA/optimizer state in `logs/ddp_locat_smoke`.
- Strict EMA/online reload and real-data backward pass with CTC weights 0 and
  0.03 pass. Both sigma and alpha predictors have finite nonzero gradients.
  See `logs/locat_checkpoint_validation.json`.
- The longest three CTC-feasible real examples have 1199, 1190 and 1188 frames.
  A padded 3×1199 bf16 CFM+CTC backward pass succeeds with an additional live
  4911 MiB reservation approximating Adam/EMA/DDP storage. Peak allocated memory
  is 21,543 MiB; this is a stress approximation, not a measured DDP peak.
  See `logs/locat_long_sequence_memory.json` and
  `src/aligndit/script/misc/stress_test_locat_memory.py`.
- One Setting-1 sample decodes successfully from step-3 EMA at two NFE, exercising
  full/TTS/null CFG and the frozen Semantic-VAE decoder. This is not a trained
  quality comparison. See `logs/locat_inference_smoke/inference_summary.json`.

The formal run starts again from the pinned S2c 70k parent, not from this smoke
checkpoint. No batch-size reduction or activation-checkpointing change was
required for the primary AV configuration. Bidirectional memory at the full
training batch has not been profiled; validate it before launching that ablation.
