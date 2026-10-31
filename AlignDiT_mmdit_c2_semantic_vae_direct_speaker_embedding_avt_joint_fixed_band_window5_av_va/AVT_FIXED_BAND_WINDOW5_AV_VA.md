# AVT joint attention with a five-frame Gaussian soft-band core

This independent snapshot copies all **247 tracked regular files** from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_av_va`
at revision `1b4b5a27b4f21fce78e715c45a4daf6d11c572fd`. Each copy was
SHA256-verified before modification. Runtime outputs and checkpoints were
excluded. The source experiment remains separate and unchanged.

## Window semantics and controlled changes

The source uses Gaussian standard deviation **0.100 seconds**, equal to
four frames on the aligned **40-Hz feature grids**. Its informal nine-frame
window describes the center plus four positions on each side within one
standard deviation. The Gaussian has no finite cutoff.

This run changes sigma to **0.050 seconds**, equal to **two feature frames**.
The corresponding one-standard-deviation core contains five positions:
the center and two positions on either side. The exact prior is:

```text
B_av[i,j] = -0.5 * ((j / 40 - i / 40 - 0.0) / 0.050)**2
B_va      = B_av.transpose(-1, -2)
```

Gaussian tails beyond ±2 remain. For offsets of 0, 1, 2 and 3 frames, the
logit biases are 0, -0.125, -0.5 and -1.125; the multiplicative prior at
±3 is exp(-1.125), approximately 0.32465. There is no hard masking at a
five-frame boundary. At a fixed nonzero distance, the negative logit bias
has four times its source magnitude. This prior is applied only to AV and
VA logits within the shared audio/video/text softmax. Same-modality and
text edges retain their source behavior.

The composed training configuration differs from the source in exactly
three fields:

- `model.arch.temporal_band_fixed_sigma_seconds`: 0.100 → 0.050;
- `model.name`: the independent `avt_joint_fixed_band_window5_av_va` name;
- `ckpts.save_dir`: the independent checkpoint directory for this run.

The Gaussian kernel, backbone, trainer and data code retain the source
implementation. The experiment identity guard requires the new run marker
and sigma 0.050. The inherited training-sidecar checks reject using sigma
0.100 weights with the sigma 0.050 inference configuration. Keep
`speaker_training_contract.json` beside checkpoints because the fixed
Gaussian constants add no parameters or tensor state.

Config:
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window5_av_va.yaml`.

## Training recipe

Retain **12 AVT MM-DiT blocks plus 6 audio-only blocks**, 376,563,842 parameters,
802 model state keys, CAM++ speaker conditioning, 64-D/40-Hz Semantic-VAE
latents and the original dataset, decoder, normalization and speaker cache.

Initialize from the same S2c **70k EMA** parent:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt`,
SHA256 `02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4`.
Start with a fresh optimizer and update counter; the inherited EMA counter
starts at 70000. The source soft-band and hard-window trained checkpoints
are not used for initialization.

Preserve seed 666; physical GPUs 0–3; BF16; 3600 frames/GPU and max 32
samples/GPU; LR 5e-5; 20k LR warmup; the **200-epoch scheduler horizon**;
CTC zero through 10k and linear warmup to 0.03 at 30k; EMA decay 0.999;
numbered saves every 50k and last saves every 5k. Training stops at
**200000 updates**, as in the source fixed-band experiment. Both 150k and
200k numbered checkpoints will be saved if training completes normally.

## Launch and TensorBoard

Run from this snapshot with the existing environment. The launchers set
the local `PYTHONPATH=src`; no editable package installation is needed.
Check GPUs 0–3 and DDP port **29642** before launch. The inherited launcher
refuses GPUs already using more than 500 MiB.

```bash
source env.sh
mkdir -p logs
TENSORBOARD_LOGDIR="$PWD/runs" bash scripts/start_speaker_tensorboard.sh
```

TensorBoard defaults to **6024**. The command above serves this snapshot's
`runs/`, including both smoke and formal runs. Only rank 0 writes events.
Check total/flow/CTC losses and gradients, and ensure
`temporal_band/sigma_ms_{min,mean,max}` are all **50**, with zero offset.
Verify the HTTP response and growing events. Open the IDE bottom **Ports**
panel, find **6024**, and click that row's actual **Forwarded Address** link.
The client forwarding address cannot be inferred from the internal server IP.

Use a separate three-update real-data validation with the original batch
size and scheduler before the formal run:

```bash
smoke_name="avt_joint_fixed_band_window5_av_va_smoke3_$(date +%Y%m%d_%H%M%S)"
smoke_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/$smoke_name"
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window5_av_va_4x4090.sh \
  "model.name=$smoke_name" "ckpts.save_dir=$smoke_dir" optim.run_until_update=3 \
  > "logs/$smoke_name.log" 2>&1 < /dev/null &
smoke_pid=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$smoke_pid"
```

Verify finite losses/gradients, actual sigma 50 ms, strict online/EMA reload,
and exit of all smoke workers. Use an empty formal checkpoint directory;
existing checkpoints trigger the inherited automatic resume behavior.

```bash
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window5_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_fixed_band_window5_av_va.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_avt_joint_fixed_band_window5_av_va.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

After each background launch, keep the launching shell alive until
`SID=PID` and `TTY=?` are confirmed, then monitor actual workers and events.

Formal checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window5_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Formal run/event directory, relative to this project:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window5_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

## Inference

The existing speaker evaluation/inference shells and Python default config
select this new soft-band experiment. Their default checkpoint is **200000**;
use `CKPT_STEP=150000` for the earlier numbered save. The protocol remains
213 CelebV-Dub Setting 1 clips, EMA, seed 0, CFG text/video 5/2 and 32 NFE.
Historical copied reports describe their source experiments, not this run.
Logs, verification reports, events, generated samples and weights are runtime
artifacts excluded from Git.
