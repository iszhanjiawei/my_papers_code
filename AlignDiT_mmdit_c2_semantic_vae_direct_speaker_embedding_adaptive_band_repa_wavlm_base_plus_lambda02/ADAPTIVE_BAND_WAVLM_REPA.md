# Adaptive temporal band + WavLM-Base+ REPA

This project is an independent source copy of the completed
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus`
implementation for a matched REPA-weight ablation.
The copy was committed before integration. It contains its own implementation;
the parent lambda-0.1 project and its running experiment are unchanged.
Training logs, checkpoints and TensorBoard events were not copied.

## Experiment contract

- Keep the parent's adaptive Gaussian bias on audio-query/video-key attention
  in all 12 MM-DiT blocks, including branch-specific video dropout and CFG behavior.
- Add the same frozen WavLM-Base+ teacher used by the existing REPA experiment:
  `microsoft/wavlm-base-plus`, revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`, final layer 12, 768 dimensions.
- Read cached features from each complete, unmasked GT utterance at 50 Hz.
  Interpolate each unpadded sequence to its valid 40-Hz latent length.
- Tap the audio output of zero-based block 9 (the 10th MM-DiT block), after its
  adaptive-band attention, and project with a `768 -> 2048 -> 2048 -> 768` SiLU MLP.
- Average `1 - cosine_similarity` over the flow-matching generation mask only;
  prompt and padding frames are excluded from the loss reduction.
- Use fixed `repa_lambda=0.2` from the first update:
  `loss = diff_loss + 0.2 * repa_loss + ctc_lambda(update) * ctc_loss`.
- Keep the parent CTC schedule: zero through 10k, then linear to 0.03 at 30k.
  Preserve seed 666, LR 5e-5, 20k LR warmup, bf16, 3,600 frames/GPU,
  200-epoch scheduler horizon and a 200k-update stop.
- Initialize from the same SHA-pinned S2c 70k EMA parent with a fresh optimizer.
  Combined strict migration expects 714 target tensors: 303 imported and 411
  new; the same ten source projector tensors are ignored. The four band tensors
  and six REPA tensors are explicitly checked.
- Sampling reconstructs both modules and loads all EMA weights strictly. The
  band remains active; the REPA projector is not called during sampling.
  SPKSIM remains WavLM-Large + ECAPA.

The primary config is:

```text
src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus.yaml
```

It inherits the adaptive-band config and overrides only the REPA settings and
experiment/output identity. The new head is constructed after existing modules,
so the same seed retains the parent's initialized tensors.

## Data and validation

The existing complete 79,613-record cache is reused read-only:

```text
/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16
```

The loader checks the pinned teacher identity, train-manifest hash, complete
coverage markers, dimensions, dtype and finite per-sample values. No WavLM
teacher is instantiated in a training worker. If preparing another machine,
the copied `src/aligndit/run/misc/extract_wavlm_base_plus_repa_celebvdub_4x4090.sh`
builds this cache.

Run from this project root, with the environment path adjusted for ROOT_PREFIX
if needed:

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_adaptive_temporal_band.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_semantic_vae_c2_repa.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_adaptive_band_repa.py
```

For a real-parent/cache BF16 forward/backward check on one available GPU:

```bash
CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_adaptive_band_repa_real_parent.py
```

This diagnostic checks real S2c migration, real latent/video/speaker/teacher
batches, the combined loss and finite nonzero gradients for the speaker,
adaptive-band output and REPA head at CTC weights 0 and 0.03. It performs no
optimizer updates and saves no training checkpoints.

## Training and inference

When launching training, start TensorBoard as well:

```bash
mkdir -p logs
bash scripts/start_adaptive_band_repa_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_adaptive_band_repa_wavlm_base_plus_4x4090.sh > logs/train_adaptive_band_repa.log 2>&1 < /dev/null &
```

The training launcher defaults to four GPUs and distributed port 29635
(`TRAIN_PORT` can override it). TensorBoard serves this project's `runs/`
on port 6015 (`TENSORBOARD_PORT` can override it). Only rank 0 writes scalars.
Existing band offset/sigma and speaker/CTC diagnostics are retained alongside
`repa_loss`, `repa_lambda`, `repa_weighted_loss`, `repa_fraction_of_total`,
`repa_projector_grad_norm` and `repa_projector_weight_norm`.

Checkpoint directory (prefixed by ROOT_PREFIX where configured):

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus_lambda02_ctc003_warmup10k30k_40hz_CelebVDub_char
```

After training produces a checkpoint:

```bash
CKPT_STEP=200000 EVAL_GPU=2 bash src/aligndit/run/eval/infer_celebvdub_s1_svae_speaker_adaptive_band_repa_wavlm_base_plus.sh
# Alternatively, use the complete generation + four-metric pipeline:
CKPT_STEP=200000 EVAL_GPU=2 bash src/aligndit/run/eval/eval_celebvdub_s1_svae_speaker_adaptive_band_repa_wavlm_base_plus.sh
```

These are alternative launch paths: both generate WAVs and reject a nonempty
output directory. The protocol remains CelebV-Dub Setting 1, 213 clips, same-clip
GT prompt, EMA, seed 0, Euler/EPSS, 32 NFE, CFG text/video 5/2 and real duration.
The copied speaker-CTC inference/evaluation launchers redirect to these entries.
The inherited adaptive-band training launcher also redirects to the combined run.

The implementation itself does not start a formal training or evaluation run.

## Parent verification and lambda-0.2 validation

- 19 inherited adaptive-band tests, 8 combined band/REPA tests, and the original
  speaker and REPA contract suites passed.
- The combined tests cover exact inference/CFG output preservation, shared
  flow/REPA masks, frozen targets, activation-checkpoint gradient parity,
  REPA-only gradients through both band output heads, and strict model/EMA reload.
- The resolved config preserves all inherited adaptive-band/data/optimizer/CTC
  fields. Training-config resolution, Python compilation and shell syntax pass.
- On physical GPU 2, the actual pinned S2c EMA checkpoint migrated exactly:
  313 source / 714 target / 303 loaded / 10 ignored / 411 new tensors.
- Two real training clips (126 and 98 latent frames) passed BF16 forward/backward
  at CTC=0 and 0.03; speaker, band offset/width and REPA gradients were finite
  and nonzero. This lambda-0.2 snapshot repeated that check successfully: its
  REPA loss was 1.00099802, weighted contribution was 0.20019960, and the
  projector gradient norm was 0.26367414. Peak allocated GPU memory was
  2.64 GiB for this small check;
  this does not estimate full-batch training memory.
- The check performed zero optimizer updates. Its successful runtime report is
  `logs/validate_adaptive_band_repa_real_parent_retry.log` (ignored by Git).
  Formal multi-GPU training and quality evaluation have not been run.
