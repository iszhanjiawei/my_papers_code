# Step 1: local audio-query/video-key attention

This independent source snapshot was copied from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding`. The original project is
unchanged. Source files are copied rather than linked across experiments; runtime
logs, checkpoints, TensorBoard events, generated samples and caches are excluded
from the copy. Existing input datasets, pretrained weights and speaker caches are
shared as read-only resources.

## Experiment definition

Here, A->V means audio supplies Query and video supplies Key/Value: video
information enters the audio stream. In the first 12 multimodal blocks:

| Query -> Key/Value | Step-1 visibility |
| --- | --- |
| Audio -> audio | Global, as in the parent |
| Audio -> video | Local: `abs(audio_index - video_index) <= 2` |
| Video -> audio | Global, as in the parent |
| Video -> video | Global, as in the parent |

The active configuration sets `model.arch.av_local_window_radius: 2`. The shared
40-Hz audio/video grid has 25-ms spacing, so radius 2 gives five accessible video
positions away from sequence boundaries, spanning +/-50 ms around the audio
position. The radius is in tokens, not seconds. Enabling it requires
`audio_video_ratio: 1`; setting it to `null` disables this structural restriction
and retains the original global joint-attention path.

Audio queries still use one joint softmax over all accessible audio and local
video keys. The change adds no gates, learned parameters or losses. It does not
close video-to-audio attention, localize video self-attention, change the text or
speaker paths, or alter the final six audio blocks.

The structural window applies during both training and inference, including when
the existing training code sets the padding mask to `None`. When an existing
padding mask is supplied, the structural window is combined with it. The parent
experiment's padding, absent-video and CFG null-video treatment remains intact;
the window does not introduce a new policy for zero-valued video positions.

At Setting-1 inference, the audio reference prefix and its zero-filled video
placeholder occupy the same shared coordinate system. The target audio and target video start at the
same offset; the local window uses these full-sequence coordinates rather than
placing target video at index zero. The inherited Setting-1 reference protocol
is unchanged. Global video attention and stacked multimodal layers still provide
indirect access to wider context: this experiment restricts direct A->V access,
not the full network's receptive field.

## Training contract and isolation

The active config retains its filename:

`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`

Both training and inference pass its complete `model.arch` to the backbone. The
training entry records the resolved config, project directory and TensorBoard
directory in `speaker_training_contract.json`; the radius therefore travels with
the run contract. The attention mask introduces no persistent state tensors, so
the strict S2c migration and existing checkpoint tensor schema remain unchanged.

This is a new optimization run from the same pinned S2c update-70,000 EMA parent,
not a continuation of the trained global-attention speaker model. Retained settings:

- Seed 666, per-rank training RNG 666 + rank.
- 79,613 CelebVDub training records; normalized 64-D/40-Hz Semantic-VAE latents.
- Frozen 192-D CAM++ inputs and zero-initialized speaker projection in blocks 12..17.
- Four GPUs, bf16, 3,600 latent frames per GPU, gradient accumulation 1.
- Peak learning rate `5e-5`, 20k LR warmup and the inherited 200-epoch decay horizon.
- CTC weight 0 through update 10k, linearly reaching 0.03 at 30k; CTC taps `[6, 12]`.
- Stop at update 200,000; numbered checkpoints every 50k, `model_last.pt` every 5k.

The run name is:

```text
AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_step1_r2
```

Checkpoint output, using the existing `ROOT_PREFIX` convention:

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_step1_r2_40hz_CelebVDub_char
```

TensorBoard events are written inside this copied project at:

```text
runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_av_local_step1_r2_semantic_vae_40hz_CelebVDub_char
```

The inherited generic and older experiment configs remain historical entry points;
use the active speaker config and launchers below for this experiment. Its default
checkpoint, Hydra, TensorBoard and evaluation outputs are isolated from the parent.

## Launching and monitoring

Run from this copied project's root. Launchers locate that root and explicitly set
`PYTHONPATH` so another editable `aligndit` installation cannot select the parent
snapshot by mistake. Do not run `pip install -e .` to switch snapshots.

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_av_local_step1_r2.log 2>&1 < /dev/null &
bash scripts/start_speaker_tensorboard.sh
```

Check that the detached training shell and its workers are running, losses remain
finite, event files grow and TensorBoard responds. `TRAIN_PORT` and
`TENSORBOARD_PORT` override service ports. Only rank 0 records TensorBoard events;
the inherited tags include total `loss`, `diff_loss`, active `ctc_loss`,
`ctc_lambda`, `ctc_weighted_loss`, `ctc_fraction_of_total`, `grad_norm/global`,
`speaker_proj_grad_norm`, `speaker_proj_weight_norm` and `lr`. The run does not add
an attention-specific loss. Runtime status, process IDs and actual forwarded URLs
must be recorded at launch rather than inferred from these example commands.

## Inference and comparison

The copied speaker inference/evaluation launchers select the new checkpoint
directory and the same active local-attention config:

```bash
bash src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh
bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

The default protocol is CelebVDub Setting 1, 213 samples, update-200k EMA, seed 0,
32 NFE, CFG text/video 5/2 and true duration. It uses the target clip's full ground
truth audio as the reference prompt; it is not an independent-reference protocol.
`CKPT_STEP`, `EVAL_GPU`, `CFG_VIDEO`, `CFG_TEXT`, `NFE` and `OUTPUT_DIR` are supported.

Compare against the global-attention speaker baseline using the same data,
initialization, training budget and inference protocol. Evaluate AVSync, WER,
SPKSIM, EMOSIM and audio quality together. This implementation only establishes
the experimental change; improved synchronization or quality is a hypothesis to
be tested, not an implementation guarantee.

## Validation before training (2026-09-21)

The copy was made from the local source at repository commit
`587824fc5ac157f1c2ad48cc8a3c4aecef01c6ba`. All 234 copied parent files were
checked against their original SHA256 values after implementation; the original
files are unchanged and the copy uses independent regular files.

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_av_local_step1.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py
```

All three checks passed. The step-1 tests cover the four attention quadrants,
direct remote-video perturbations and gradients, unchanged parameter keys,
wide-window equivalence to global attention, training with absent padding masks,
activation-checkpoint recomputation, CFG, prefix coordinates and sampling.
The real-parent check validates the pinned parent and dataset, loads 303 of 704
target state keys through the inherited strict migration, and runs finite
forward/backward passes at CTC weights 0 and 0.03 without updating parameters.
The resolved Hydra config differs from the parent only in model name,
checkpoint output directory and `av_local_window_radius`.


## Equivalent split-query implementation (2026-09-21)

Local attention now splits audio and video query rows into two SDPA calls. Both
calls retain the concatenated audio/video keys and values. Audio uses an additive
0/-infinity mask with the same radius; video uses the inherited padding mask (or
no mask in training). This is the same row-wise softmax as the original dense
joint-square boolean mask. No model tensors, loss, schedule or visibility change.
Floating-point kernel differences mean bitwise-identical training is not promised.

Validation adds an independent float64 dense reference for output, input-gradient
and parameter-gradient comparisons, radii 0/2/20, and padded/unpadded batches.
The existing locality, CFM/CTC, checkpoint and CFG tests also pass. Run:

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_av_local_step1.py
CUDA_VISIBLE_DEVICES=0 /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/benchmark_av_local_query_split.py
```

On this machine, isolated bf16 SDPA tests at B=8/T=400 and B=4/T=800 gave
output relative RMS differences of 0.163%/0.168%, and Q/K/V-gradient differences
of 0.21%-0.33%. These are numerical tensor comparisons, not quality metrics.
The old implementation uses memory-efficient SDPA; the new video branch uses
Flash SDPA. Isolated attention forward/backward changed from 1.050 to 1.176 ms
and 1.725 to 1.625 ms respectively: query splitting alone does not establish a
consistent throughput gain. End-to-end training speed must be measured separately;
the historical fixed-band rate of 2.52 updates/s is not guaranteed on this host.

The user authorized abandoning the initial unsaved run and starting again from
the same pinned 70k EMA parent with seed 666. Old logs and TensorBoard events are
archived outside the active TensorBoard run before restarting, so step numbers
from independent runs cannot overlap in the active loss curve.
