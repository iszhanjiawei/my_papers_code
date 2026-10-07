# Semantic-VAE Direct-C2 + CAM++ speaker embedding

This independent source snapshot was copied from `AlignDiT_mmdit_c2_semantic_vae_direct`.
Speaker conditioning is ported from `AlignDiT_mmdit_c2_speaker_embedding`; neither source
experiment is modified. The copy includes the parent's current local source changes.
Logs, data, checkpoints, generated audio and TensorBoard events are not copied or committed.

## Experiment

- Retain normalized 64-D / 40-Hz Semantic-VAE latents, aligned 40-Hz video, all 79,613
  training records, CTC taps `[6, 12]`, and CTC strides `[1, 1]`.
- Read frozen bilingual CAM++ embeddings from the existing complete raw-audio cache.
  Each vector is a finite, L2-normalized `float32[192]`; no speaker encoder is trained.
- Project with zero-initialized, bias-free `Linear(192,768)` (147,456 added parameters).
  Add the projection to timestep conditioning only in zero-based blocks 12 through 17.
  The first 12 multimodal blocks and output timestep conditioning retain the base path.
- Drop speaker and prompt audio together. Full/TTS CFG branches keep speaker; the null
  branch removes both speaker and prompt. Batched CFG preserves branch-major mask order.
- Start a new optimization run from the same pinned S2c 70k EMA parent as Direct-C2.
  Strict migration loads 303 tensors, ignores the same 10 S2c HuBERT tensors, and leaves
  401 new tensors (400 original C2 tensors plus speaker projection) at initialization.
- Peak learning rate `5e-5`, original 20k LR warmup and 200-epoch decay horizon, gradient
  clipping 1.0, bf16, 3,600 latent frames per GPU, 4 GPUs. CTC is 0 through update 10k,
  ramps linearly to 0.03 at 30k, and remains 0.03 thereafter.
- Stop after update 200,000 without shortening the inherited LR scheduler horizon.
  Save numbered checkpoints every 50k and `model_last.pt` every 5k.
- Record initialization seed 666; per-rank training RNG uses 666 + rank. The older
  lambda=0.03 entry did not explicitly seed model initialization, so the historical
  comparison is not a strictly paired same-initialization experiment.

## Training

Configuration:

`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`

It inherits the Direct-C2 lambda=0.03 config, which inherits the base VAE config.
The training entry is `src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker.py`.

From this project directory:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_speaker_ctc003.log 2>&1 < /dev/null &
bash scripts/start_speaker_tensorboard.sh
```

Checkpoint directory, under the existing `ROOT_PREFIX` convention:

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Only rank 0 writes TensorBoard. The run directory is:

```text
runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

Scalars include `loss`, `diff_loss`, `ctc_loss` once enabled, `ctc_lambda`,
`ctc_weighted_loss`, `ctc_fraction_of_total`, `grad_norm/global`,
`speaker_proj_grad_norm`, `speaker_proj_weight_norm`, and `lr`. Losses retain the
base trainer's rank-0 batch reporting convention. No new speaker loss is added.
The TensorBoard launcher uses port 6006 by default (`TENSORBOARD_PORT` overrides it).

## Validation and inference

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/audit_semantic_vae_speaker_cache.py --full-audit
bash src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh
bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

Inference launchers support `CKPT_STEP`, `EVAL_GPU`, `CFG_VIDEO`, `CFG_TEXT`, `NFE`,
and `OUTPUT_DIR`. The default is the 200k EMA, CFG video 2.0, CFG text 5.0 and NFE 32.
Speaker vectors come from the original waveform corresponding to the inference prompt,
not from a VAE reconstruction. Existing CelebVDub Setting 1 uses the same GT clip as
prompt and target; this protocol is retained and recorded in the inference summary.
These scores must not be described as evaluation with an independent reference clip.

The existing speaker cache is read-only. To prepare one on a different server, the
copied extraction launcher is `src/aligndit/run/misc/extract_campplus_celebvdub_4x4090.sh`.
Training and inference check cache metadata, encoder identity, dimensions and coverage;
every consumed vector is validated. `speaker_training_contract.json` and Hydra's
resolved config in the new checkpoint directory document each run.
