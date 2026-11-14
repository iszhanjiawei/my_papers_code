# Fixed 13-frame AVT + LibriTTS initialization on GRID

This is an independent copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va_libritts_pretrained`
at Git `5da427000355bed377d94c2dae333f44a00bcf4c`. The byte-identical source copy
was committed as `32c7ad48219fc960c43c5f1c1e7e8ff1a31d09b3`; logs, weights,
TensorBoard events and generated outputs were not copied. The original project
and the GRID reference project remain unchanged.

The GRID reference is
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus_grid`.
Only its data protocol, dataset validation and launch/schedule checks are reused.
The new model has neither its adaptive temporal predictor nor its REPA head/loss.
Inherited CelebV-Dub/LRS3 launchers and evaluation documents describe the source
experiments; use the GRID entry below for this snapshot.

## Model and initialization

- 12 AVT joint-attention blocks + 6 audio-only blocks, unchanged backbone.
- Fixed AV/VA Gaussian attention bias: sigma 0.150 s, offset 0, aligned 40-Hz
  features. The nominal core is +/-6 frames (13 total); tails remain soft.
- Full EMA initialization from
  `${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/libriSpeech_LibriTTS_pretrained_models/model_00050000.pt`.
  SHA256: `4ee7dd9e1377728a042e75ebbbe70c2a12cd830c9b078afe13ebca289d03503c`.
- Preserve all 802 transferred tensors, including trained text and CAM++ speaker
  projection parameters. The same five vocabulary-dependent tensors are mapped
  by character/filler/CTC-blank identity to the unchanged 159-character vocabulary.
- Fresh optimizer, scheduler and GRID update counter starting at zero. Carry the
  parent EMA step 50000, matching the source experiment. No CelebV-Dub-trained
  checkpoint is used for initialization.

## Data and schedule

Cache: `${ROOT_PREFIX}/zjw524/projects/data/GRID_svae1000k_sample_seed666_fp32`.
It contains 29,531 train and 3,274 held-out val utterances; original train/val
assignments and the existing 163/26 exclusions are preserved. The held-out split
is not fed into training and there is no online validation loop.

The cache uses complete MPG audio aligned to the mouth-frame timeline, 16-kHz
mono, 64-D FP32 Semantic-VAE posterior samples, 1024-D FP32 AV-HuBERT video
interpolated to the same 40-Hz length, and 192-D normalized CAM++ embeddings.
The pretraining vocabulary and LibriSpeech normalization remain unchanged.
Existing REPA cache files are unused. Features are not regenerated.

GRID's extraction metadata and completed inventory are validated before model
creation. Initialization checks the GRID metadata's actual codec, normalization
and posterior-sampling protocol instead of treating its copied source VAE spec
as a GRID extraction receipt. Per-sample reads check dimensions, finite values
and speaker-vector normalization.

- Four RTX 4090 GPUs, BF16, 3,600 frames/GPU, max 32 samples/GPU, seed 666.
- Learning rate 5e-5, 20k LR warmup, gradient accumulation 1, max grad norm 1.
- CTC weight zero through update 10k, linear increase to 0.03 at update 30k.
- Same GRID schedule as the reference: 1,000 epochs, stop at update 200,000.
  The actual sampler has 985 frame batches, 246 per rank on four GPUs, dropping
  one tail batch after each deterministic epoch shuffle. Capacity is 246,000
  updates; the inherited 200 epochs would only allow 49,200. This changes the
  LR decay horizon relative to the CelebV-Dub run and is recorded explicitly.
- Numbered checkpoints every 50k updates; latest checkpoint every 5k.

## Start training

From this project directory:

```bash
source env.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_grid.sh \
  > logs/train_grid_window13_libritts_pretrained.log 2>&1 < /dev/null &
train_pid=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

The entry checks the requested GPU count, full train cache, batch schedule and
parent checkpoint hash, starts a separate TensorBoard service, then execs
Accelerate. Default DDP port is 29644 (`TRAIN_PORT` override). TensorBoard chooses
a free port starting at 6028 (`TENSORBOARD_PORT` override), and reuses only a
verified service belonging to this exact snapshot. `NUM_PROCESSES`,
`CUDA_VISIBLE_DEVICES` and `PYTHON_BIN` are explicit runtime overrides.

Use an empty checkpoint directory for the first run. Later launches resume only
a matching GRID training contract; inherited or differently configured weights
in the same directory are rejected.

Checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_GRID_ctc003_warmup10k30k_40hz_char
```

TensorBoard run/logdir relative to this project:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_GRID_ctc003_warmup10k30k_semantic_vae_40hz_GRID_char
```

The service scans this project's `runs/` directory. Its actual PID, port and
local URL are written to `logs/tensorboard_grid.json`; launch/process metadata
are recorded in `logs/grid_training_run.json`. Open the IDE bottom panel's
Ports tab and click the actual forwarded address for that port. A local/private
IP URL is not a verified IDE forwarding URL.

The checkpoint directory records `speaker_training_contract.json` (resolved
configuration, GRID hashes, schedule and fixed-band geometry) and
`parent_migration.json` (actual parent and all tensor/vocabulary mappings).
Runtime artifacts remain untracked.

## Verification

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  scripts/test_grid_transfer.py --real-parent
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/check_grid_training.py --num-processes 4
```

The tests check preserved model/trainer implementations and configuration,
train/val separation, actual batch schedule, sample collation, rejected invalid
contracts, and exact full-model LibriTTS EMA transfer on CPU. They perform no
optimizer updates. Launch validation additionally checks real workers, fresh
training loss and gradient scalars, and TensorBoard HTTP/scalar visibility.
