# AVT joint attention with a strict thirteen-frame A↔V window

This independent snapshot copies all 256 tracked files from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window5_av_va`
at commit `a70d77b73857b24a04ee07169925f83a5f82136e`. Copies are regular files,
SHA256-verified before editing. Runtime outputs and checkpoints were excluded.
The source experiment remains unchanged.

## Controlled changes

The radius changes from 2 to **6** on the aligned **40-Hz feature grids**:

```text
B_av[i,j] = 0       if abs(j - i) <= 6
B_av[i,j] = -inf    otherwise
B_va      = B_av.transpose(-1, -2)
```

Each interior query permits thirteen cross-modal keys: the center and six on
each side (±150 ms). Sequence boundaries clip the window to existing keys.
Direct AV and VA attention probabilities outside the window are exactly zero;
within-window logits receive no distance penalty. AA, VV and every text edge
retain their original behavior, with joint normalization over all A/V/T keys.
The inherited Gaussian sigma is inactive in strict-window mode.

The new run stops at **150000 updates**. `optim.epochs=200`, the LR scheduler
horizon and warmup retain the source values. This is early stopping of the same
training recipe: it does not shorten or rescale the learning-rate schedule.
The existing save/stop order writes `model_150000.pt` and `model_last.pt` before
exit. Numbered saves at 50k and 100k are also retained.

The resolved configuration differs from hard5 in exactly four fields:

- `model.arch.temporal_band_fixed_window_radius_frames`: 2 → 6;
- `optim.run_until_update`: 200000 → 150000;
- `model.name`: the independent hard13 run name;
- `ckpts.save_dir`: the independent hard13 checkpoint directory.

Config:
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window13_av_va.yaml`.
Inherited configuration files live within this snapshot. Backbone, temporal
mask implementation, trainer, data and objectives retain the source code.
The experiment identity guard now requires radius 6 and the hard13 marker.
The training sidecar records radius 6/width 13 and prevents mismatched window
checkpoints from being used silently.

## Initialization and training recipe

Start from the same S2c **70k EMA** parent:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt`,
SHA256 `02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4`.
Training optimizer/update counters start from zero; the inherited EMA internal
counter starts at 70000. Use an empty new checkpoint directory on the first
launch, because existing matching weights automatically trigger resume.

Retain 12 AVT MM-DiT blocks plus 6 audio-only blocks; CAM++ speaker conditioning;
Semantic-VAE 64-D/40-Hz latents; four GPUs 0–3; BF16; seed 666; 3600 frames/GPU;
up to 32 samples/GPU; LR 5e-5; 20k LR warmup; CTC zero through 10k and linearly
raised to 0.03 at 30k; EMA decay 0.999; saves every 50k and last every 5k.
Dataset, decoder, normalization, speaker cache and padding policy are inherited.

## Launch and monitoring

Run from this snapshot root with the existing environment. Each launcher
sets this project's `PYTHONPATH=src`; no editable-package change is needed.
Check GPUs 0–3 and DDP port **29641** before launch. TensorBoard defaults to
**6023** and refuses an occupied port.

```bash
source env.sh
mkdir -p logs
TENSORBOARD_LOGDIR="$PWD/runs" bash scripts/start_speaker_tensorboard.sh
```

Run a separate three-update real-data validation with original batch/scheduler:

```bash
smoke_name="avt_joint_hard_window13_av_va_smoke3_$(date +%Y%m%d_%H%M%S)"
smoke_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/$smoke_name"
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window13_av_va_4x4090.sh \
  "model.name=$smoke_name" "ckpts.save_dir=$smoke_dir" optim.run_until_update=3 \
  > "logs/$smoke_name.log" 2>&1 < /dev/null &
smoke_pid=$!
printf '%s\n' "$smoke_pid" > "logs/$smoke_name.pid"
```

Confirm a detached session, finite real losses/gradients, TensorBoard radius 6
and width 13, and successful strict online/EMA checkpoint reload. Wait for all
smoke workers to exit. Then launch the formal run with no overrides:

```bash
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window13_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_hard_window13_av_va.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_avt_joint_hard_window13_av_va.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

For each background launch, keep the launching shell alive until `ps` confirms
`SID=PID` and `TTY=?`; this avoids exiting while a cold-loaded `setsid` binary
has not detached yet. Monitor the workers and live loss/event stream.

Formal checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window13_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Formal run/event directory, relative to this project:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window13_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

TensorBoard serves this project's `runs/` so smoke and formal curves are both
available. Only rank 0 writes events. Verify the HTTP response and actual
loss/flow/CTC scalar values, as well as radius 6 and width 13. Open the IDE's
bottom Ports panel, locate port 6023 and click its displayed forwarded-address
link. The client's actual forwarding URL cannot be inferred from an internal
server address.

## Inference and validation

The existing speaker inference/evaluation shell entries select the hard13
configuration and isolated output directories, defaulting to **150000**.
The inference Python default config also selects hard13. Keep the matching
`speaker_training_contract.json` beside weights. Evaluation protocol retains
213 CelebV-Dub S1 clips, EMA, seed 0, CFG text/video 5/2 and 32 NFE.

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_avt_hard_window13_av_va.py --cuda
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_avt_hard_window13_contract.py
```

Other inherited experiment/evaluation reports describe their historical source
runs. Copy hashes, configuration audits, validation logs, TensorBoard events
and launch status are runtime artifacts kept outside Git under `logs/`,
`runs/` and the independent checkpoint directories.
