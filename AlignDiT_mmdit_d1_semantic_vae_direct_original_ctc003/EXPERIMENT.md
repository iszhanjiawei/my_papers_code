# Original D1 + Semantic-VAE, CTC 0.03 with 10k->30k warmup

This is an independent source copy of the restored original
`AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` (restoration commit `f1fc9f2`).
The source-copy commit is `40cef80`. Neither that project nor the existing
`AlignDiT_mmdit_d1_semantic_vae_direct` warmup experiment is modified.
Historical logs, results, datasets and checkpoints were not copied.

On 2026-09-07, the user requested stopping the fixed-CTC run and adding warmup.
This project directory keeps its original name, but the current config/run uses
`Warmup10k30k`. The fixed run's artifacts are preserved in their old directories.

## Architecture and training contract

- Original D1: 6 multimodal blocks + 12 inherited audio-only DiT blocks.
- Audio/video joint attention followed by **Audio-only** text cross-attention.
  No Hunyuan dual-stream text CA, CA RoPE or all-head RoPE is introduced.
- Self-attention RoPE remains on head 1; Q/K RMSNorm is retained.
- CTC heads remain at zero-based block indices `[5, 11]`.
- `ctc_lambda = 0.03` is the target: updates 1..10000 use zero CTC;
  updates 10001..30000 ramp linearly to 0.03; later updates retain 0.03.
  The 20k learning-rate warmup is independent and unchanged.
- CelebVDub: all 79,613 training examples, including 105 CTC-infeasible examples
  whose CTC loss is zeroed by the inherited `zero_infinity=True` behavior.
- Fixed Semantic-VAE posterior-sample cache: 64D, 40 Hz, 16 kHz / hop 400;
  fixed LibriSpeech-train channel normalization, not re-estimated on CelebVDub.
- Cached lip features are interpolated from 25 to 40 Hz. Audio/video ratio is 1
  and CTC strides are `[1, 1]`. This is rate conversion, not text/phoneme alignment.
  No forced alignment or transcript timestamp preprocessing is added.
- Strict initialization from the S2c-70k Semantic-VAE pure-audio **EMA** parent
  used by the reference C2 project. Source 313 tensors, target 559, loaded 303,
  10 auxiliary-projector tensors ignored, 256 new multimodal/CTC tensors.
  The original 80D mel parent cannot directly initialize the 64D frontend/output.
- Preserved: 768 width / 18 blocks / 12 heads, AdamW LR `5e-5`, LR warmup 20k,
  200 epochs, EMA 0.999, accumulation 1, max samples 32, workers 16 per rank,
  seed 666, `checkpoint_activations=False`.
- Necessary frame-budget conversion: 3600 at 40 Hz = 90 s/GPU, matching original
  D1's 9000 at 100 Hz. Four GPUs, bf16, no activation checkpointing.
- `log_samples=False` follows the reference VAE training entry: the inherited
  sample logger is mel-vocoder-only. Dedicated VAE inference below decodes
  generated latents after reversing normalization with the pinned VAE decoder.
- Saves: `model_last.pt` every 5k, numbered checkpoints every 50k.

The `cfm_vt.py` and audio-only DiT implementations remain unchanged.
Core model changes are the identity frontend and configurable CTC sampling
ratios (mel default `[2, 1]`). The trainer now calls an optional pre-forward
schedule hook and writes diagnostics; its optimizer and LR schedule are unchanged.

## Entry points

Use this snapshot as working directory and always set `PYTHONPATH=src`.
Do not run `pip install -e .` in the shared environment. Historical copied
configs/scripts are retained for provenance; they are **not** this experiment.

Configuration:
`src/aligndit/config/finetune_celebvdub_mm_d1_semantic_vae_direct.yaml`.

```bash
# CPU architecture / CTC schedule, resume contract and gradient checks
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_direct.py

# Full-size bf16 forward/backward, strict parent migration and latent sampling
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_d1_direct.py --real-data --device cuda

# Start detached four-GPU training and TensorBoard; prints PIDs and logs
bash scripts/start_d1_svae_ctc003_warmup.sh

# After a checkpoint exists: inference and four historical Setting-1 metrics
setsid env PYTHONUNBUFFERED=1 bash \
  src/aligndit/run/eval/eval_celebvdub_s1_d1_semantic_vae.sh 150000 \
  > logs/eval_150k.log 2>&1 &
```

The launch helper requires a new checkpoint directory by default. To intentionally
resume this exact run, use `RESUME=1`; the inherited trainer restores the child
optimizer, scheduler and update, not the S2c parent's optimizer/update.
Do not start a second process group on top of a running job.

## Artifact locations

Run: `AlignDiT_MMDiT_D1_SemanticVAE_Original_CTC003_Warmup10k30k_semantic_vae_40hz_CelebVDub_char`.

TensorBoard: `runs/` followed by that run name. Loss tags are `loss`, `diff_loss`,
`ctc_lambda`, `ctc_active`, `ctc_weighted_loss`, plus `lr`; only global rank 0 writes
events. Raw `ctc_loss` starts when CTC becomes active after 10k. During the first
10k it is not evaluated: weighted CTC=0 and total loss equals diffusion loss.

Checkpoints:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_D1_SemanticVAE_Original_CTC003_Warmup10k30k_40hz_CelebVDub_char`.
`parent_migration.json` records the verified parent, loaded counts and new/ignored parameter names;
Hydra saves the resolved configuration in its timestamped `outputs/` directory.
`ctc_schedule.json` pins the schedule; incompatible/missing contracts cannot resume
existing weights. This restart loads S2c-70k EMA afresh and begins child update 1.

TensorBoard defaults to port 6006, and DDP rendezvous to 29593. Open the client
bottom panel's Ports tab and use the **actual forwarded address** for port 6006.
The launcher prints a server-local address, not a claim that client forwarding
has already been configured. Runtime logs/events/checkpoints are not committed.

## Historical fixed-CTC launch — stopped on user request

The run launched at 2026-09-07 06:21 CST was stopped around update 2881 before
the first 5k checkpoint. No child checkpoint existed. Logs, events, Hydra config
and `parent_migration.json` remain in the old `CTC003_Fixed` directories.
Its launcher, four workers and TensorBoard service have all exited.

Implementation commit: `0b5b585` (pushed to `origin/main`).

- Training launcher PID 11245; workers 11388/11389/11390/11391.
- TensorBoard PID 11244, port 6006. All service/worker TTYs are `?` and have
  independent sessions. Launcher and TensorBoard have been reparented to PID 1.
- Training log: `logs/train_20260907_062120.log`.
- TensorBoard log: `logs/tensorboard_20260907_062120.log`.
- Confirmed worker cwd points to this new snapshot, not the source project.
- Original mel/MM-DiT regression smoke passed. New fixed-CTC CPU/CUDA checks
  passed. Full-size real-data bf16 loss/gradients and latent sampling passed;
  the 303 migrated parent tensors were bit-identical.
- Bound VAE decoder test passed: 40 latent frames decode to 16,000 finite samples.
- By update 63, all four loss/LR tags were available through TensorBoard HTTP.
  Across those steps, maximum discrepancy from `diff_loss + 0.03 * ctc_loss`
  was below `1e-7`. Early throughput was 3.34 updates/s (not a long-run estimate).
- TensorBoard root and Scalars API returned HTTP 200. The client-side forwarded
  URL is not exposed to the available tools; it was requested from the user and
  must still be verified separately. No forwarded URL is fabricated here.

These are launch checks, not a convergence or generated-speech quality claim.
