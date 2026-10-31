> Historical source-experiment record. For the current five-frame Gaussian soft-band run,
> see [AVT_FIXED_BAND_WINDOW5_AV_VA.md](AVT_FIXED_BAND_WINDOW5_AV_VA.md).

# AVT joint attention with fixed bidirectional audio/video bands

This isolated source copy starts from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint` and adds the
fixed audio/video prior from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_fixed_band_av_va`.
The source projects are retained as separate experiments.

## Attention rule

The first 12 MM-DiT blocks still perform one joint softmax over audio, video and
text. Add the following Gaussian logit prior only to the audio-query/video-key
and video-query/audio-key submatrices:

```text
B_av[i,j] = -0.5 * ((j / 40 - i / 40 - 0.0) / 0.100)**2
B_va      = B_av.transpose(-1, -2)

              audio keys  video keys  text keys
 audio query      0          B_av         0
 video query      B_va        0            0
 text query       0          0            0
```

The offset is 0 seconds and sigma is 0.100 seconds on the aligned 40-Hz grids.
Sigma is the Gaussian standard deviation, not a hard cutoff or total window
width. These constants are shared across samples, heads, layers and training
updates, including dropped-video CFG branches. Text has no temporal penalty;
all modalities retain the same joint normalization. The prior has no trainable
parameters, new state tensors, new loss or content-dependent predictor.

The following 6 audio-only blocks, CAM++ speaker injection at blocks 12–17,
Semantic-VAE 64-channel latents, text stream, CTC taps and padding policy retain
the source AVT behavior. This change does not by itself establish a measured
quality or synchronization improvement.

## Training and initialization

Use `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_av_va.yaml`.
It inherits the original AVT speaker config and changes only the band settings,
experiment name and output directory. The original YAML remains available for
regression checks. The dedicated launcher and generic speaker launcher both
default to this new configuration.

Initialize compatible weights from the same SHA-pinned S2c **70k EMA** parent:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt`.
A fresh run starts with a new optimizer and update counter. The completed AVT
150k/200k checkpoints are not the initialization for this controlled experiment.
Retain seed 666, 4 GPUs, BF16, 3,600 frames/GPU, max 32 samples/GPU, LR 5e-5,
20k LR warmup, the original 200-epoch decay horizon and the stop at 200k updates.
CTC is zero through 10k and ramps linearly to 0.03 at 30k. Numbered checkpoints
are saved every 50k, and `model_last.pt` every 5k. The original 79,613 training
records, including CTC-infeasible examples, remain in the training split.

Run from this snapshot's root:

```bash
source env.sh
mkdir -p logs
bash scripts/start_speaker_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_fixed_band_av_va.log 2>&1 < /dev/null &
train_pid=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

The launcher checks physical GPUs 0–3 and rejects a GPU already using more than
500 MiB. Default DDP port: **29636** (`TRAIN_PORT` override). Check it is free
before starting. Existing checkpoints in the configured save directory are
automatically resumed, so use an empty directory for the initial formal run.
Keep the generated `speaker_training_contract.json` with the checkpoints;
fixed attention constants have no tensor state and are recorded in this contract.

Checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

TensorBoard event directory, relative to this project:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

TensorBoard uses **6020** by default (`TENSORBOARD_PORT` override). The startup
script serves the exact formal event directory; `TENSORBOARD_LOGDIR` can select
another run. Only rank 0 writes total/flow/CTC losses, learning rate, weighted
CTC, existing speaker/global gradient diagnostics and fixed-band offset/sigma
statistics. After startup, verify
workers, finite progressing losses, event updates, TensorBoard HTTP response
and scalar tags. Use the application's Ports panel to obtain the actual
forwarded URL for the current port.

A real-data smoke run must use a separate `model.name` and `ckpts.save_dir`,
with `optim.run_until_update=3`. The trainer saves `model_last.pt` when it reaches
this limit; retain the original save intervals, full batch size and 200-epoch scheduler.
For smoke curves, start TensorBoard with `TENSORBOARD_LOGDIR="$PWD/runs"` to
serve both smoke and formal event directories. Confirm all smoke workers exit
before starting the formal run.

## Validation (2026-10-24)

- All 14 fixed-band attention tests pass, including independent explicit joint
  softmax equivalence and CUDA BF16 checkpointed backward. Disabling the band
  matches the original AVT initialization, RNG, state and outputs exactly.
- All 11 training/inference contract tests pass. The inherited AVT regression
  suite passes its 9 CPU tests (its optional CUDA test was skipped).
- Full-size models have the same 376,563,842 trainable parameters, 802 state
  keys and shapes, 12 AVT MM-DiT blocks and 6 audio-only blocks. No parameter or
  state tensor is added by the prior.
- A separate 4-GPU real-data run, `avt_joint_fixed_band_av_va_smoke3_20261024`,
  completes 3 updates with the original 3,600-frame batch per GPU. Losses are
  1.3477653, 1.4531589 and 1.3474160; speaker/global gradients are finite.
  TensorBoard reports offset 0 ms and sigma 100 ms at all 3 steps, and
  `model_last.pt` is saved at update 3. Both online and EMA weights reload
  strictly with all 802 model keys and finite tensors. EMA retains the
  inherited step policy (parent 70,000 + 3 new updates = 70,003); the new
  training update counter is 3. Runtime verification reports are in `logs/`
  and are excluded from version control.

## Inference and evaluation

The copied entries
`src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh` and
`src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh` now select
the new fixed-band AVT config, checkpoint directory and output directory.
They retain Setting 1, same-clip reference, EMA, seed 0, CFG text/video 5/2 and
32 NFE, with the existing `CKPT_STEP`, `EVAL_GPU`, `CFG_VIDEO`, `CFG_TEXT`,
`NFE` and `OUTPUT_DIR` overrides. The default checkpoint step is 200000.
`scripts/audit_avt_joint_s1.py` accepts explicit output directories and has no
hard-coded checkpoint directory.

Other copied configs, named upstream launchers and historical evaluation
reports describe their original experiments; use the entries above for this
run. Logs, events, samples and checkpoint files are runtime artifacts and are
not committed.
