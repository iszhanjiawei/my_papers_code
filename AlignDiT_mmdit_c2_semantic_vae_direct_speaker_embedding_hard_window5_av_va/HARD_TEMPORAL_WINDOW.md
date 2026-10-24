# Isolated five-frame hard AV+VA temporal window

This experiment is an independent source copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_fixed_band_av_va`
from repository revision `d81985cc4f19414ca636b440e81d099987e15b14`.
The source snapshot is unchanged. Source files are real copies, without shared
code links. Existing training artifacts were not copied; the pinned datasets,
frozen codec, speaker caches and S2c parent remain read-only shared inputs.

## Attention rule

For the first 12 multimodal blocks, at aligned 40-Hz model-input frame indices:

```text
B_av[i,j] = 0       if abs(i-j) <= 2
            -inf   otherwise
B_va = B_av.transpose(-1, -2)
joint bias = [[0_AA, B_av], [B_va, 0_VV]]
```

An interior query can directly attend to exactly five opposite-modality
positions: `i-2, i-1, i, i+1, i+2`. Out-of-window AV/VA attention weights are
exactly zero. At sequence boundaries the window is clipped to existing keys;
it is not shifted or padded. Integer indices avoid floating-point boundary
errors. These are 40-Hz feature frames (25 ms each), so the extreme permitted
timestamps are 50 ms before and after the query. Original 25-fps video frames
are interpolated by the existing preprocessing before this mask is applied.

Inside the window, the usual learned query/key scores and joint softmax
determine attention weights. There is no Gaussian weighting. The inherited
sigma field remains for interface compatibility and has no effect on this
hard mask; TensorBoard reports `window_radius_frames=2` and
`window_size_frames=5` instead of sigma diagnostics. AA/VV and text attention
retain their original behavior. The five-frame constraint applies to direct
cross-modal attention in each block, not the network's full receptive field.

The hard window is active in both training and inference, including dropped
video CFG branches and training with padding-attention masking disabled. It
adds no parameters or checkpoint tensors. A sidecar training contract records
the radius, kernel and direction; inference rejects a Gaussian checkpoint
paired with a hard-window config, or the reverse.

## Configuration and training

Use `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_hard_window5_av_va.yaml`.
It inherits the source AV+VA config and changes the experiment/output names and
`temporal_band_fixed_window_radius_frames: 2`. Legacy named training wrappers
in this copy are disabled to prevent writing into other experiments.

The matched protocol initializes from the same SHA-pinned S2c 70k EMA parent
with a fresh optimizer and counter: seed 666, four GPUs, BF16, 3,600 latent
frames/GPU, LR 5e-5, 20k LR warmup, 200-epoch scheduler horizon, CTC zero through
10k and linearly increased to 0.03 at 30k, stopping at 200k updates. Numbered
checkpoints are saved every 50k updates and `model_last.pt` every 5k.

Run from this directory:

```bash
source env.sh
bash scripts/start_hard_window5_av_va_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_hard_window5_av_va_4x4090.sh \
  > logs/train_hard_window5_av_va.log 2>&1 < /dev/null &
train_pid=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Default rendezvous port: 29631. TensorBoard port: 6012. Both can be overridden
when occupied. Checkpoints are isolated under `${ROOT_PREFIX}` at:

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_hard_window5_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

TensorBoard serves this copy's `runs/`. The formal event directory is:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_hard_window5_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

Open port 6012 in the IDE's Ports panel and use its actual forwarded URL.
`http://127.0.0.1:6012` is the server-local service address; client access
requires a corresponding forwarded port. Logs, weights, events and runtime
validation reports are ignored by Git.

## Validation

CPU regression entry points (run with `PYTHONPATH=src` and the standard
`${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python` interpreter):

```text
src/aligndit/script/misc/smoke_test_hard_temporal_window.py
src/aligndit/script/misc/smoke_test_hard_window_contract.py
src/aligndit/script/misc/smoke_test_fixed_temporal_band_bidirectional.py
src/aligndit/script/misc/smoke_test_fixed_temporal_band.py
src/aligndit/script/misc/smoke_test_adaptive_temporal_band.py
src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
```

A real-data smoke uses a unique `hard_window5_av_va` name/directory and three
updates via `optim.run_until_update=3 ckpts.save_per_updates=3
ckpts.last_per_updates=3`, retaining the full scheduler horizon. Then use
`src/aligndit/script/misc/validate_fixed_band_checkpoint.py` with that checkpoint
and `--step 3` to verify strict online/EMA loading and finite gradients at CTC
weights 0 and 0.03. The copied speaker inference/evaluation wrappers default to
this new config and checkpoint directory.

`FIXED_TEMPORAL_BAND.md`, `ADAPTIVE_TEMPORAL_BAND.md` and other inherited
experiment notes describe historical snapshots, not this experiment's active
attention rule or launch commands.

### Verified on 2026-10-25 (Asia/Shanghai)

- 15 hard-window tests, 60 inherited attention tests, four speaker checks and
  four configuration/checkpoint/diagnostic suites passed. Fully masked padding
  rows now use a discarded same-modality self-key to avoid PyTorch 2.4 SDPA NaN
  gradients; no cross-modal exclusion is relaxed.
- Four-GPU BF16 smoke `hard_window5_av_va_smoke_20261025` completed three updates
  with losses 1.34673667, 1.41288972 and 1.36197352. Strict S2c migration matched
  313 source / 704 target / 303 loaded / 10 ignored / 401 new tensors.
- The saved checkpoint passed strict EMA and online loading. Real-data backward
  checks at CTC weights 0 and 0.03 had finite global gradient norms 1.10872352
  and 1.26567543. A one-item, two-NFE inference produced 17,462 finite samples
  at 16 kHz. These checks establish functionality, not generation quality.
- TensorBoard's Scalars API exposed total/diffusion/weighted-CTC losses and
  constant radius 2 / size 5. Client-side forwarding URLs are not exposed to
  the execution session; server-local HTTP availability was verified.
- Original-source SHA256 checks matched all 253 copied files, with no shared
  source inodes. Reports are in `logs/` and
  `ckpts/hard_window5_av_va_smoke_20261025/`; live launch details are recorded in
  `logs/launch_status_20261025.json`.
