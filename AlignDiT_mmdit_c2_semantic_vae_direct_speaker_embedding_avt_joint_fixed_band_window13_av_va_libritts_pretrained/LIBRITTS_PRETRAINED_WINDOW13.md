# Fixed 13-frame AVT experiment with LibriTTS 50k EMA initialization

This independent snapshot copies the historical fixed-window13 experiment at
Git `82bd7b1`; its source-only copy is recorded by `e34a81d`. Runtime logs,
checkpoints and TensorBoard events were not copied. Inherited evaluation reports
refer to earlier experiments, not to this new run.

The user requested the same fixed13 downstream experiment initialized from:

```text
${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/libriSpeech_LibriTTS_pretrained_models/model_00050000.pt
```

This is a speech/text joint LibriTTS checkpoint, not the earlier audio-only S2c
checkpoint. Load its full EMA model state, including the trained text and speaker
parameters. The new loader validates its exact identity and adapts vocabulary
rows while retaining the original downstream vocabulary and architecture.

## Preserved downstream configuration

- Fixed Gaussian AV/VA sigma 0.150 seconds: +/-6 aligned 40-Hz frames, nominal
  13-frame soft core; Gaussian tails remain, and there is no hard window.
- 12 AVT joint blocks and 6 audio-only blocks; original text padding behavior,
  CAM++ conditioning, dataset/cache, Semantic-VAE codec and latent normalization.
- 4 GPUs, BF16, 3,600 frames/GPU, at most 32 samples/GPU, seed 666.
- LR 5e-5, 20k LR warmup, 200-epoch scheduler, stop at downstream update 200k.
- CTC zero through 10k, increasing linearly to 0.03 at 30k.
- Numbered checkpoints every 50k and last checkpoint every 5k.

Only initialization and the experiment's output identities change. Optimizer,
scheduler and downstream update count start fresh. The inherited EMA policy
retains the parent's completed EMA step, now 50000, while downstream training
starts at update 0. No LibriTTS optimizer or scheduler state is resumed.

## Verified parent and vocabulary mapping

Parent size: `5048750348` bytes.
Parent SHA256: `4ee7dd9e1377728a042e75ebbbe70c2a12cd830c9b078afe13ebca289d03503c`.
Embedded training-contract SHA256:
`88e16bfe6a0c393bd4236e70a1bebd824713cc00c06bbb709e3a530d433f236f`.
The checkpoint has schema 1, stage `libritts_text_audio`, update 50000 and
initialized EMA step 50000. Its embedded contract is checked directly; it is not
represented as an S2c checkpoint with an unrelated external contract.

The source vocabulary has 171 characters; the downstream vocabulary retains
159. The source preparation preserves all original character indices and appends
12 sorted characters: `(`, `)`, `/`, `:`, `;`, `[`, `]`, `{`, `}`, `ê`, `œ`, and
combining macron U+0304. Reconstructing that vocabulary from the verified target
vocabulary reproduces its pinned source SHA256 exactly:
`7032586c60c7d17d17e0d580f8cdb05b5652326e43a287f405367dcacfc74ac0`.
The target remains
`225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d`.

All 802 target tensor names are present in the source EMA. 797 have identical
shapes. The remaining five are mapped by character identity:

- Text embedding: retain filler row 0; character row is token index + 1.
- Both final CTC weights/biases: character row uses the unshifted token index;
  source unused row 171 maps to target row 159, and source blank row 172 maps to
  target blank row 160. Do not truncate away the correct source blank row.

Full-state initialization also transfers the source's visual and CTC tensors;
those branches were inactive during LibriTTS pretraining and are not claimed to
have learned visual or CTC supervision. Source pretraining used text padding
masking, while this downstream run deliberately retains the historical fixed13
`text_mask_padding=False`, as requested. Source-only mode and activation
checkpointing settings do not replace the original downstream settings.

## Launch and artifacts

Configuration:
`src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained.yaml`.

From this project directory:

```bash
source env.sh
mkdir -p logs
bash scripts/start_speaker_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_4x4090.sh \
  > logs/train_speaker_window13_libritts_pretrained.log 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > logs/train_speaker_window13_libritts_pretrained.pid
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

Default DDP port: 29642 (`TRAIN_PORT` override).
Default TensorBoard port: 6026 (`TENSORBOARD_PORT` override).

Checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_ctc003_warmup10k30k_40hz_CelebVDub_char
```

TensorBoard logdir relative to this project:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

Start the initial run with an empty checkpoint directory. The existing trainer
resumes matching downstream checkpoints if that directory is reused. The
`parent_migration.json` receipt records the actual EMA transfer and row mapping;
`speaker_training_contract.json` records the resolved downstream configuration.
The copied speaker inference/evaluation entries default to this experiment's
config and checkpoint directory. Validation logs and run receipts stay in the
ignored `logs/` directory.

## Validation before training

The dedicated parent test passed all nine checks, including actual-checkpoint
SHA verification, all 802 online/EMA tensors compared exactly, filler/CTC blank
mapping, rejected wrong identity/contract/update, unchanged downstream settings
and core model/training implementation, and an empty optimizer with update 0.
No optimizer update is performed by these tests.

```bash
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_libritts_parent.py --real-parent
```

The inherited fixed-band contract suite passed all 11 checks, and all five
changed/new shell entry points passed `bash -n`. The resolved configuration
audit permits differences only in parent identity, initialization mode,
experiment name and checkpoint output path.
