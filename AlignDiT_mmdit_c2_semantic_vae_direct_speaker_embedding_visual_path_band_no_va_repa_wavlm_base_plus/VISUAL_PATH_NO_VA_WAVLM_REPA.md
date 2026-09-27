# Visual Path Band + blocked VA + WavLM-Base+ REPA

This project is an independent source copy of the Visual Path Band experiment.
It replaces the preceding local-AV/no-VA/REPA experiment's hard AV window with
the original Visual Path Band soft prior. The original Visual Path Band,
local-AV/no-VA/REPA and adaptive-band REPA snapshots remain unchanged.

## Attention and supervision

The first 12 MM-DiT blocks use the following topology:

| Region | Behavior |
| --- | --- |
| Audio queries / audio keys (AA) | Global |
| Audio queries / video keys (AV) | Original Visual Path Band soft bias |
| Video queries / audio keys (VA) | Blocked |
| Video queries / video keys (VV) | Global |

Audio queries retain a single shared softmax over audio and video keys. Video
queries attend directly to video keys when VA is blocked. There is no radius-2
hard AV mask, visual-specific gate, local VV mask or trainable band predictor.
The existing AdaLN residual modulation remains unchanged. The video branch is
still trainable through the audio objectives.

The AV logit bias is unchanged from the parent:

```text
B[i,j] = -0.5 * ((t_video[j] - t_audio[i]) / 0.100 seconds)^2
         -0.5 * ((c40[j] - c40[i]) / 2.0)^2
```

The scalar path is accumulated from L2 increments of unit-normalized native
25-Hz frozen AV-HuBERT video features, then interpolated to the valid 40-Hz grid
with `align_corners=False`. Keep offset 0, time sigma 100 ms and path sigma 2.0.
The native and interpolated feature caches are reused read-only. The original
branch-specific masking is retained: remove edges touching hidden/padded frames,
reaccumulate the path, and use a zero path for null-video CFG. The S1 dummy
prompt uses a flat path and the target is rebased without a fictitious jump.

REPA matches the previous WavLM-Base+ experiments:

- Frozen `microsoft/wavlm-base-plus`, revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`, teacher layer 12, 768 dimensions.
- Reuse all 79,613 cached complete GT utterances from
  `${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16`.
  Each unpadded 50-Hz target is interpolated to its valid 40-Hz audio length.
- Tap zero-based block 9 (the 10th MM-DiT block); use a
  `768 -> 2048 -> 2048 -> 768` SiLU projection head initialized last.
- Compute detached-teacher `1 - cosine_similarity` only over flow-generation
  frames; prompt and padding do not contribute.
- `loss = diff_loss + 0.1 * repa_loss + ctc_lambda(update) * ctc_loss`.
  No teacher network is loaded by a training worker.
- Inference loads the projector strictly but does not call it or require
  teacher features. Native video paths remain required.

The VA block, temporal/path constants and REPA tap index cannot all be checked
through tensor keys. Keep `speaker_training_contract.json` beside checkpoints;
inference verifies the recorded topology, band geometry, path source and REPA
configuration before loading EMA weights.

## Validation before launch (2026-09-28)

- 97 CPU unit tests passed: 10 combined attention/path tests, 12 REPA and
  inference-contract tests, and the inherited 31 Visual Path Band, 25 fixed-band
  and 19 adaptive-band regression tests. The generic REPA smoke also passed.
- Split attention matches an explicit full attention matrix with the AV soft
  prior and VA blocked, in outputs and gradients. Tests include rectangular
  lengths, empty video masks, training, CFG, path masking and checkpointing.
  Distant AV remains softly weighted rather than prohibited by a hard window.
- Same-seed inherited weights are unchanged; only the six REPA projector
  tensors are added. Teacher gradients, padding and prompt loss exclusions,
  strict model/EMA reload and mismatched inference contracts were checked.
- Compared resolved configs against the Visual Path Band parent: exactly 13
  intended leaf differences (VA switch, REPA parameters/metadata and identity);
  the original data, optimizer, schedule and path geometry are unchanged.
- A real-parent, real-native-video, real-WavLM-cache BF16 forward/backward check
  passed on physical GPU 0. Migration loaded 303 exact S2c tensors into 710
  target tensors (407 new, 10 ignored source tensors).
- The two clips have 126/98 latent frames and 156/121 teacher frames. Native
  path coordinates were nonzero. At CTC weights 0 and 0.03, total losses were
  1.599807 and 1.879622; REPA loss was 0.996373. Video, speaker and REPA gradients
  were finite and nonzero. No optimizer updates were performed. The small check
  peaked at 2.64 GiB, which is not the formal-batch memory requirement.

Repeat the combined checks from this project directory:

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_visual_path_no_va.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_visual_path_no_va_repa.py
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_visual_path_no_va_repa_real_parent.py
```

The GPU command is a zero-update integration check; select an available GPU.
Validation logs are in the ignored `logs/` directory. These checks establish
implementation behavior, not generation quality or improved AVSync.

## Training protocol and entry points

Initialize from the same SHA-pinned S2c 70k EMA parent with a fresh optimizer;
do not continue a trained Visual Path Band or prior REPA run. Preserve seed 666,
4 GPUs, bf16, 3,600 frames/GPU, max 32 samples/GPU, LR 5e-5, 20k LR warmup,
200-epoch LR horizon and stop at 200k updates. CTC remains zero through 10k,
then increases linearly to 0.03 at 30k. Save numbered checkpoints every 50k
updates and `model_last.pt` every 5k.

Config:

```text
src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_visual_path_band_no_va_repa_wavlm_base_plus.yaml
```

Launch from this project root:

```bash
mkdir -p logs
bash scripts/start_visual_path_band_no_va_repa_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_visual_path_band_no_va_repa_wavlm_base_plus_4x4090.sh > logs/train_visual_path_band_no_va_repa.log 2>&1 < /dev/null &
```

The launcher uses GPUs 0/1/2/3 and port 29636 (`TRAIN_PORT` override).
TensorBoard serves this project's `runs/` directory on port 6016 by default
(`TENSORBOARD_PORT` override). Only rank 0 writes total, flow, CTC and REPA
scalars, projection gradients, and the inherited temporal/visual-path diagnostics.
Record actual process IDs, event logdir and verified addresses per server in
the ignored `logs/launch_status.json`; do not reuse another server's PIDs/URLs.

Checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_visual_path_band_no_va_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char
```

TensorBoard run (under this project's `runs/`):

```text
AlignDiT_MMDiT_c2_svae_speaker_visual_path_band_no_va_repa_wavlm_base_plus_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

The copied speaker and Visual Path Band launchers redirect to this combined run.
Other copied upstream entries/documents are historical references.

After checkpoints exist, choose either inference or the full evaluation pipeline:

```bash
CKPT_STEP=150000 EVAL_GPU=0 bash src/aligndit/run/eval/infer_celebvdub_s1_svae_speaker_visual_path_band_no_va_repa_wavlm_base_plus.sh
CKPT_STEP=200000 EVAL_GPU=0 bash src/aligndit/run/eval/eval_celebvdub_s1_svae_speaker_visual_path_band_no_va_repa_wavlm_base_plus.sh
```

The full pipeline includes generation and refuses existing nonempty outputs.
The benchmark protocol stays Setting 1, 213 clips, same-clip GT reference,
EMA, seed 0, Euler/EPSS, 32 NFE, text/video CFG 5/2 and true duration.
The combined experiment does not isolate the independent effect of VA blocking
or REPA; those require matched ablations.
