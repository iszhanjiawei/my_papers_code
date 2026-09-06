# Original D1 + Semantic-VAE + speaker embedding, CTC warmup 10k->30k

This is an isolated source copy of
`AlignDiT_mmdit_d1_semantic_vae_direct_original_ctc003` at `9503f14`.
The unmodified copy is committed as `524e496`; speaker implementation was
initially added in `b649aa3` before any training launch. The source D1 was then
updated in `ff0f14a` to CTC warmup (documented at `376ca78`). This speaker
snapshot now inherits that current schedule rather than its initial fixed-CTC
copy. The old fixed-speaker version was never launched as a training run.
Neither the original D1 project
nor the reference `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding` is
modified. Logs, events, results, checkpoints and data were not copied.
There are no shared-source symlinks or changes to the environment's editable install.

## What changed

- Frozen CAM++ speaker vectors, following the C2 speaker implementation:
  `iic/speech_campplus_sv_zh_en_16k-common_advanced`, 192D FP32, L2 normalized.
  The encoder is not loaded or optimized during diffusion training.
- A **zero-initialized, bias-free Linear(192, 768)** adds a speaker delta to the
  timestep condition of D1's twelve audio-only blocks, zero-based **6..17**.
  C2 used 12..17 because its multimodal prefix is longer. No speaker delta
  is added to D1's multimodal blocks 0..5 or the final output normalization.
  This adds exactly 147,456 trainable parameters / one state-dict tensor.
- The initial speaker delta is zero, preserving the original initialized
  computation. Parent audio weights remain eligible for strict migration.
- Training speaker dropout follows the final audio-prompt dropout decision.
  CFG keeps speaker identity in conditional and TTS branches and removes it
  in the unconditional branch; `no_ref_audio` drops speaker conditioning too.
- Dataset and trainer pass the cached vector to CFM, and the dedicated
  Semantic-VAE inference entry retrieves the vector of the **reference audio**.
  Training vectors come from each complete, unmasked training waveform,
  exactly as in the reference C2 implementation; no forced alignment is added.
- CFG packed inference repeats times/masks in branch-major order, matching
  concatenated features and speaker conditions for batches larger than one.
  This does not alter training or historical batch-one evaluation.
- TensorBoard adds speaker weight/gradient norms and CTC-weight diagnostics;
  finite-loss/gradient checks fail loudly if the new path becomes invalid.

## What did NOT change from the source D1

- Original D1: **6 multimodal + 12 native audio-only DiT** blocks; width 768,
  depth 18, 12 attention heads. Audio/video joint attention followed by
  **Audio-only** text cross-attention, original Q/K RMSNorm, first-head RoPE.
  No Hunyuan dual-stream text CA, CA RoPE or all-head RoPE.
- CTC taps `[5, 11]`, 40-Hz sampling strides `[1, 1]`; the current source
  D1's CTC warmup is preserved: child updates 1..10000 have weight **0**,
  updates 10001..30000 ramp linearly to **0.03**, and later retain **0.03**.
  The 20k learning-rate warmup is independent and unchanged.
  At zero CTC weight, raw CTC loss is not evaluated and CTC heads do not train;
  speaker conditioning is still trained through the diffusion loss.
- All 79,613 CelebVDub training records, including the original 105
  CTC-infeasible records handled by `zero_infinity=True`.
- Fixed Semantic-VAE posterior-sample cache: 64D / 40 Hz / 16 kHz / hop 400;
  pinned LibriSpeech normalization. Lip features are interpolated 25 to 40 Hz;
  audio/video ratio 1. This is rate conversion, not transcript alignment.
- Same **S2c-70k Semantic-VAE pure-audio EMA** initialization, with checkpoint
  size/SHA256 and parent-contract validation. Child optimizer/update start
  fresh. Migration: source 313, target **560**, loaded **303**, ignored 10,
  new **257** (source D1 had 559/256; the only extra key is speaker projection).
- AdamW LR `5e-5`, LR warmup 20k, 200 epochs, EMA 0.999, accumulation 1,
  3600 latent frames/GPU, max samples 32, workers 16 per rank, seed 666.
  Four GPUs, bf16, **checkpoint_activations=False**.
- `log_samples=False`: the inherited logger is mel-vocoder-only. The dedicated
  VAE inference entry performs inverse normalization and VAE decoding.
- `model_last.pt` every 5k, numbered checkpoints every 50k. No new
  stop-at-200k override is imported from C2; the original 200-epoch schedule remains.

The config diff against the current source D1 at `376ca78` consists only of four speaker-cache fields,
two speaker architecture fields, and the isolated model/run name.

## Entry points

Work in this snapshot and set `PYTHONPATH=src`. Do not `pip install -e .`.
Copied historical configs/scripts remain for provenance; they are not this run.
The active config name stays `finetune_celebvdub_mm_d1_semantic_vae_direct`.

```bash
# CPU speaker parity, layer isolation, CFG/dropout, and gradient checks
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_speaker.py

# Read-only validation of every training speaker vector
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/audit_semantic_vae_speaker_cache.py --full-audit

# Full-size real-data bf16 forward/backward, strict parent migration, sampling
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_direct.py --real-data --device cuda

# Detached training on GPUs 0..3 and an independent TensorBoard service
bash scripts/start_d1_svae_ctc003_warmup.sh

# After a child checkpoint exists: Setting-1 inference and four metrics
setsid env PYTHONUNBUFFERED=1 bash \
  src/aligndit/run/eval/eval_celebvdub_s1_d1_semantic_vae.sh 150000 \
  > logs/eval_150k.log 2>&1 < /dev/null &
```

The training helper rejects an existing checkpoint directory unless `RESUME=1`
is explicitly set. Resume restores this child run's optimizer/scheduler/update.
Never launch a second group while this experiment is already running.

## Runtime locations

Run name:
`AlignDiT_MMDiT_D1_SemanticVAE_Original_CTC003_Warmup10k30k_Speaker_semantic_vae_40hz_CelebVDub_char`.

TensorBoard logdir: this project's `runs/` followed by the exact run name.
Only global rank 0 writes events. Tags: `loss`, `diff_loss`, `ctc_loss`, `lr`,
`ctc_lambda`, `ctc_active`, `ctc_weighted_loss`, `ctc_fraction_of_total`, `grad_norm`,
`speaker_proj_weight_norm`, `speaker_proj_grad_norm`.

Checkpoints:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_D1_SemanticVAE_Original_CTC003_Warmup10k30k_Speaker_40hz_CelebVDub_char`.
`parent_migration.json` records strict migration; `speaker_training_contract.json`
records speaker/config/init semantics; `ctc_schedule.json` pins the warmup for
resume (incompatible or missing contracts cannot reuse existing checkpoints).
Hydra saves a resolved config under `outputs/`.

Speaker cache (read-only reuse):
`${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/campplus_spk_emb_zh_en_16k`.

Default TensorBoard port **6007**, DDP port **29594**, both checked before launch.
Open the client's Ports panel, locate 6007, and click its actual forwarded
address. A server-local URL is not a verified client forwarding URL.
No forwarding URL is fabricated in this document.

Runtime logs, events, caches, generated samples and weights are not committed.

## Launch verification — 2026-09-07 06:44 CST

- Active implementation commit: `b0b2afe` (pushed to origin/main). This includes
  the source D1 warmup update, not the earlier unlaunched fixed-CTC copy.
- Launcher PID **19396**, workers **19511/19512/19514/19516**. TensorBoard PID
  **19395**, port **6007**. Launcher/TensorBoard are reparented to PID 1;
  workers and services have independent sessions and TTY `?`.
- Training log: `logs/train_20260907_064438.log`.
  TensorBoard log: `logs/tensorboard_20260907_064438.log`.
  Worker cwd and the single rank-0 event file belong to this speaker snapshot.
- All 79,613 train and 213 test speaker vectors passed shape/dtype/finiteness/
  unit-norm checks; original batch features remain identical without the added
  field. Audit: `logs/speaker_cache_audit.json`.
- CPU warmup boundaries/resume-contract/CFG/speaker-gradient checks passed.
  Real-data full-size CUDA bf16 checks passed at CTC weights 0, 0.015 and 0.03,
  including nonzero speaker gradients and a finite two-step latent sample.
  The 303 inherited parent tensors were bit-identical after migration.
  Log: `logs/smoke_warmup_real_data_prelaunch.log`.
- Live migration confirms source 313 / target 560 / loaded 303 / new 257,
  parent EMA update 70k, fresh child update 1; `ctc_schedule.json` confirms
  target 0.03 with boundaries 10000/30000.
- At the update-65 dashboard check, all ten active scalar tags were finite;
  `ctc_lambda`, `ctc_active`, `ctc_weighted_loss` were exactly zero, and total
  loss exactly matched diffusion loss. Speaker projection norm increased from
  zero to 0.0005290203, with finite/nonzero gradients on conditioned updates.
  Raw `ctc_loss` is intentionally absent until CTC becomes active after 10k.
- Four GPUs were computing with approximately 11.3–11.8 GiB each at an early
  snapshot, with activation checkpointing OFF. Startup throughput around
  3 updates/s is not a long-run ETA or convergence claim.
- TensorBoard root and Scalars API returned HTTP 200. Exact logdir is this
  project's `runs/AlignDiT_MMDiT_D1_SemanticVAE_Original_CTC003_Warmup10k30k_Speaker_semantic_vae_40hz_CelebVDub_char`.
  Server-local address: `http://127.0.0.1:6007`. Client forwarding is not exposed
  to the available tools; the actual Ports-panel URL was requested from the
  user and remains unverified. Do not substitute an invented forwarded URL.
