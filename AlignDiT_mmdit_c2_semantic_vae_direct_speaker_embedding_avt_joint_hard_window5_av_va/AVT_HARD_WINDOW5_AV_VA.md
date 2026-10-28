# AVT joint attention with a strict five-frame A↔V window

This experiment is an independent copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window9_av_va`
at commit `4317480fcebbe9b852bed208403cfe935c0d60fa`. All 254 tracked source
files were copied as regular files and SHA256-verified before modification.
The original snapshot is unchanged. Runtime outputs were not copied.

## Controlled change

Only the temporal radius changes from 4 to 2 on the aligned 40-Hz grids:

```text
B_av[i,j] = 0       if abs(j - i) <= 2
B_av[i,j] = -inf    otherwise
B_va      = B_av.transpose(-1, -2)
```

Each interior query permits five cross-modal keys: the center, two before and
two after (±50 ms). Edges are clipped to existing valid positions. Direct AV
and VA attention probabilities outside the window are exactly zero. AA, VV
and every text edge retain their original attention behavior, and each query
still normalizes jointly over all audio/video/text keys. The inherited sigma
of 0.100 seconds is inactive in strict-window mode.

The backbone, mask implementation, trainer, objectives and parameter structure
are unchanged: 12 AVT MM-DiT blocks plus 6 audio-only blocks, CAM++ conditioning,
Semantic-VAE 40-Hz latents and the original padding policy. The training identity
guard now requires radius 2 and the `avt_joint_hard_window5_av_va` marker.
Sidecars record width 5 and reject mismatching hard9/Gaussian checkpoints.

The resolved configuration differs from the source only in:

- `model.arch.temporal_band_fixed_window_radius_frames`: 4 → 2;
- `model.name`: the independent hard5 experiment name;
- `ckpts.save_dir`: the independent hard5 checkpoint directory.

Configuration:
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window5_av_va.yaml`.
Its inherited configurations live inside this snapshot, without cross-project
source symlinks or editable-package changes. Training and evaluation set
`PYTHONPATH=src` in this project's root.

## Initialization and unchanged training recipe

A fresh run loads the same pinned S2c 70k EMA parent:
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_SemanticVAE_mel_warmstart_s2c_40hz_LibriSpeech/model_70000.pt`,
SHA256 `02e35cf3e0de2a10573fb6efd8e5b7cdf0c59a18ea07807f34e5c7bf9c1395c4`.
Optimizer and training updates start from zero; the inherited EMA internal step
counter starts at 70000. The new formal checkpoint directory must be empty on
first launch because existing matching checkpoints automatically trigger resume.

All training settings retain their source values: GPUs 0–3, BF16, seed 666,
3600 frames/GPU, at most 32 samples/GPU, LR 5e-5, 20k LR warmup, the original
200-epoch scheduler horizon, stop at 200k updates, CTC zero through 10k then
linearly raised to 0.03 at 30k, EMA decay 0.999, numbered checkpoints every 50k
and `model_last.pt` every 5k. Dataset, decoder, normalization and speaker cache
remain the same. No source-project checkpoint is resumed.

## Training and TensorBoard

Run commands from this project's root. The launcher verifies GPUs 0–3 are idle
and uses the existing Python environment directly. DDP defaults to port 29639;
TensorBoard defaults to port 6022. Ports only separate services.

```bash
source env.sh
mkdir -p logs
TENSORBOARD_LOGDIR="$PWD/runs" bash scripts/start_speaker_tensorboard.sh
```

Separate three-update real-data smoke run, with the original batch/scheduler:

```bash
smoke_name="avt_joint_hard_window5_av_va_smoke3_$(date +%Y%m%d_%H%M%S)"
smoke_dir="${ROOT_PREFIX}/zjw524/projects/data/ckpts/$smoke_name"
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window5_av_va_4x4090.sh \
  "model.name=$smoke_name" "ckpts.save_dir=$smoke_dir" optim.run_until_update=3 \
  > "logs/$smoke_name.log" 2>&1 < /dev/null &
smoke_pid=$!
printf '%s\n' "$smoke_pid" > "logs/$smoke_name.pid"
```

After checking finite real losses, gradients, window diagnostics, TensorBoard
and the update-3 checkpoint, wait for all smoke workers to exit. Launch the
formal run with no configuration overrides:

```bash
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window5_av_va_4x4090.sh \
  > logs/train_speaker_avt_joint_hard_window5_av_va.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_avt_joint_hard_window5_av_va.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Formal checkpoints:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window5_av_va_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Formal run and event directory (relative to this project):

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_hard_window5_av_va_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

The service above serves this project's `runs/` for both smoke and formal runs.
Only rank 0 writes events. Verify actual total/flow/CTC loss tags, ongoing event
updates, radius 2, width 5, worker/GPU activity and the TensorBoard HTTP response.
Open the IDE's bottom Ports panel, locate port 6022, and click its actual
forwarded-address link. The forwarded URL is assigned by the client and must
not be inferred from an internal machine address.

## Inference and checks

The existing speaker entries now select the hard5 config, checkpoint directory
and output names:

```text
src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh
src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

The evaluation protocol remains unchanged. Keep `speaker_training_contract.json`
beside weights, since the mask radius has no serialized tensor parameters.
Inherited evaluation reports describe their named historical experiments.

Validation commands:

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_avt_hard_window5_av_va.py --cuda
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_avt_hard_window5_contract.py
```

Copy hashes, configuration audits and launch status are stored in ignored
`logs/`. Logs, events, samples and checkpoints are runtime artifacts excluded
from Git.
