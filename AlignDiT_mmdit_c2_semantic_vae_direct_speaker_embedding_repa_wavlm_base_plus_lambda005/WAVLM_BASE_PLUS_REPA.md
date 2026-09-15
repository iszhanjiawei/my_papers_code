# Single-teacher WavLM-Base+ REPA — lambda 0.05

This is an isolated training variant of the existing Semantic-VAE Direct-C2 +
CAM++ speaker experiment. It adds representation alignment for generated speech
without changing the inference graph, CAM++ conditioning, CTC, or SPKSIM evaluator.

Copied from `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_repa_wavlm_base_plus`
at commit `c02703c857c492d7e3bf8d2fec30917009684663`. The only training-hyperparameter
change is REPA weight `0.1 -> 0.05`; run/checkpoint names and launch ports are
isolated. This is a fresh run from the same S2c 70k EMA parent, not a continuation
of the lambda-0.1 run. Existing teacher features are reused read-only; logs,
events and checkpoints from the source experiment are not copied.

## Fixed experiment contract

- Teacher: `microsoft/wavlm-base-plus`, repository revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`, checkpoint SHA256
  `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51`.
- Target: final transformer output, layer 12, `float16[T_50Hz,768]`, extracted
  from the complete unmasked 16-kHz ground-truth utterance. The teacher is never
  instantiated by a training worker.
- Student tap: zero-based layer 9, the output of the 10th of 12 double-stream
  MM-DiT blocks. A trainable `768 -> 2048 -> 2048 -> 768` SiLU MLP projects it.
- Alignment: each unpadded WavLM sequence is linearly resampled from 50 Hz to the
  exact valid 40-Hz Semantic-VAE length. `1 - cosine_similarity` is averaged only
  over the same random generation mask used by flow matching; prompt and padding
  frames do not enter the REPA reduction.
- Weight: `repa_lambda=0.05`. Existing diffusion loss and the delayed CTC schedule
  (0 through 10k, linear to 0.03 at 30k) are unchanged.
- Initialization: the same pinned S2c 70k EMA parent. Strict migration permits
  exactly the six new projector tensors in addition to the existing CAM++ tensor.
- Evaluation: SPKSIM remains the independent frozen WavLM-Large + ECAPA-TDNN
  pipeline in `src/f5_tts/eval/utils_eval.py`. No WavLM-Base+ REPA target or
  projector is used during sampling or metric calculation.

The primary config is:

```text
src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_repa_wavlm_base_plus.yaml
```

## 1. Build the immutable teacher cache

Run once before training:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/misc/extract_wavlm_base_plus_repa_celebvdub_4x4090.sh \
  > logs/extract_wavlm_base_plus_repa.log 2>&1 < /dev/null &
```

The launcher uses four GPUs and mirrors each manifest path under:

```text
/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16
```

Extraction is resumable at file granularity. It pins and hashes the teacher,
validates every feature, and writes `metadata.json` only after all 79,613 train
records pass `coverage_report.json`. Training refuses a missing, partial,
wrong-revision, wrong-dimensional, or wrong-manifest cache.

## 2. Validate without data or checkpoints

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_repa.py
```

The CPU smoke test verifies that the auxiliary head leaves inference output
unchanged, taps the requested block, excludes prompt frames, handles 50-to-40-Hz
alignment, receives a first backward gradient, and works with activation
checkpointing.

## 3. Train

After the cache is complete:

```bash
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_repa_wavlm_base_plus_4x4090.sh \
  > logs/train_speaker_repa_wavlm_base_plus.log 2>&1 < /dev/null &
bash scripts/start_speaker_tensorboard.sh
```

The checkpoint directory is:

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_repa_wavlm_base_plus_lambda005_40hz_CelebVDub_char
```

TensorBoard adds `repa_loss`, `repa_lambda`, `repa_weighted_loss`,
`repa_fraction_of_total`, `repa_projector_grad_norm`, and
`repa_projector_weight_norm` to the existing diffusion, CTC, speaker-projection,
gradient and learning-rate scalars.

The training target is 200,000 updates with seed 666, unchanged from the source.
TensorBoard defaults to port 6008 and this exact run-relative log directory:

```text
runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_repa_wavlm_base_plus_lambda005_semantic_vae_40hz_CelebVDub_char
```

## 4. Inference and four-metric evaluation

The REPA projector is reconstructed from the training config so its EMA weights
load strictly, but it is not called during sampling. After a checkpoint exists:

```bash
bash src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_repa_wavlm_base_plus.sh
bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_repa_wavlm_base_plus.sh
```

The four-metric wrapper retains the existing same-clip S1 protocol and uses the
same WavLM-Large + ECAPA checkpoint for SPKSIM, enabling a direct comparison
against the non-REPA CAM++ run.
