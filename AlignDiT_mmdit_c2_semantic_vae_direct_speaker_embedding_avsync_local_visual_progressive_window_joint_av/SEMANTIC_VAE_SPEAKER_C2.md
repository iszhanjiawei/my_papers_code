# Semantic-VAE Direct-C2 + CAM++ speaker embedding

This independent source snapshot was copied from `AlignDiT_mmdit_c2_semantic_vae_direct`.
Speaker conditioning is ported from `AlignDiT_mmdit_c2_speaker_embedding`; neither source
experiment is modified. The copy includes the parent's current local source changes.
Logs, data, checkpoints, generated audio and TensorBoard events are not copied or committed.

## Experiment

- Retain normalized 64-D / 40-Hz Semantic-VAE latents, aligned 40-Hz video, all 79,613
  training records, CTC taps `[6, 12]`, and CTC strides `[1, 1]`.
- Read frozen bilingual CAM++ embeddings from the existing complete raw-audio cache.
  Each vector is a finite, L2-normalized `float32[192]`; no speaker encoder is trained.
- Project with zero-initialized, bias-free `Linear(192,768)` (147,456 added parameters).
  Add the projection to timestep conditioning only in zero-based blocks 12 through 17.
  The first 12 multimodal blocks and output timestep conditioning retain the base path.
- Drop speaker and prompt audio together. Full/TTS CFG branches keep speaker; the null
  branch removes both speaker and prompt. Batched CFG preserves branch-major mask order.
- Start a new optimization run from the same pinned S2c 70k EMA parent as Direct-C2.
  Strict migration loads 303 tensors, ignores the same 10 S2c HuBERT tensors, and leaves
  401 new tensors (400 original C2 tensors plus speaker projection) at initialization.
- Peak learning rate `5e-5`, original 20k LR warmup and 200-epoch decay horizon, gradient
  clipping 1.0, bf16, 3,600 latent frames per GPU, 4 GPUs. CTC is 0 through update 10k,
  ramps linearly to 0.03 at 30k, and remains 0.03 thereafter.
- Stop after update 200,000 without shortening the inherited LR scheduler horizon.
  Save numbered checkpoints every 50k and `model_last.pt` every 5k.
- Record initialization seed 666; per-rank training RNG uses 666 + rank. The older
  lambda=0.03 entry did not explicitly seed model initialization, so the historical
  comparison is not a strictly paired same-initialization experiment.

## Training

Configuration:

`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`

It inherits the Direct-C2 lambda=0.03 config, which inherits the base VAE config.
The training entry is `src/aligndit/script/train/finetune_semantic_vae_c2_direct_speaker.py`.

From this project directory:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_speaker_ctc003.log 2>&1 < /dev/null &
bash scripts/start_speaker_tensorboard.sh
```

Checkpoint directory, under the existing `ROOT_PREFIX` convention:

```text
/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Only rank 0 writes TensorBoard. The run directory is:

```text
runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char
```

Scalars include `loss`, `diff_loss`, `ctc_loss` once enabled, `ctc_lambda`,
`ctc_weighted_loss`, `ctc_fraction_of_total`, `grad_norm/global`,
`speaker_proj_grad_norm`, `speaker_proj_weight_norm`, and `lr`. Losses retain the
base trainer's rank-0 batch reporting convention. No new speaker loss is added.
The TensorBoard launcher uses port 6006 by default (`TENSORBOARD_PORT` overrides it).

## Validation and inference

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/audit_semantic_vae_speaker_cache.py --full-audit
bash src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_ctc003.sh
bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

Inference launchers support `CKPT_STEP`, `EVAL_GPU`, `CFG_VIDEO`, `CFG_TEXT`, `NFE`,
and `OUTPUT_DIR`. The default is the 200k EMA, CFG video 2.0, CFG text 5.0 and NFE 32.
Speaker vectors come from the original waveform corresponding to the inference prompt,
not from a VAE reconstruction. Existing CelebVDub Setting 1 uses the same GT clip as
prompt and target; this protocol is retained and recorded in the inference summary.
These scores must not be described as evaluation with an independent reference clip.

The existing speaker cache is read-only. To prepare one on a different server, the
copied extraction launcher is `src/aligndit/run/misc/extract_campplus_celebvdub_4x4090.sh`.
Training and inference check cache metadata, encoder identity, dimensions and coverage;
every consumed vector is validated. `speaker_training_contract.json` and Hydra's
resolved config in the new checkpoint directory document each run.

## Audio-tail local visual attention experiment

This experiment now lives in the independent sibling project
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avsync_local_visual`.
The original `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding` project
is restored byte-for-byte to the tracked source at commit `80f5b36`.
The new project preserves the implementation from `a473ce1`, its external
checkpoint directory, and its training configuration. Project relocation
updates only `project_dir` and `tensorboard_logdir` in the runtime speaker
contract; the original contract is archived beside it. Training resumes from
`model_last.pt` with model, EMA, optimizer and scheduler states. TensorBoard
purges scalar events after the saved update before appending the resumed run.

The optional `audio_local_visual_attention` variant adds six independent modules
to zero-based blocks **12–17**. Each audio block runs self-attention, then the
new gated visual cross-attention, then its existing FFN. Blocks 0–11 retain
their original joint attention. Speaker modulation and text/CTC policy are
unchanged. Old configs default to this option being disabled.

Queries come from the current audio hidden states; keys/values come directly
from the cached 1024-D AV-HuBERT visual features on the 40 Hz timeline. They do
not pass through the global Conformer or joint attention before this adapter.
This is an intentional adaptation: Flowley's implementation uses projected
visual states after its multimodal stream. There is no added text attention or
extra synchronization loss in this experiment.

The window follows Flowley's `configs/train.yaml` and
`flowley/model/modules/layers/attention.py`: omega=0, delta=4 at 8 FPS, with a
cosine fade. Its physical radius is **0.5 seconds**, or **20 tokens at 40 Hz**.
For an audio query aligned to visual index c, the weight is
`0.5 * (1 + cos(pi * abs(j-c) / 20))` within that radius and zero outside;
the attention-logit bias is `log(weight + 1e-6)`. Thus this is a soft window,
with epsilon leakage outside it, matching Flowley's code. Only invalid keys
use `-inf`. Centers use `round`, following the reference code rather than the
paper's `floor` notation; valid lengths, not padded batch lengths, bound them.
All six layers use fade scale 1. Flowley's layer schedule changes fade strength,
not its window radius; a progressive schedule is left for a separate experiment.

The residual is `audio + attention_output * gate`. Following OmniShow's
`diffsynth/modules/gated_local_context_attention.py`, every layer has a learnable
768-D gate initialized to **1e-5**, with no sigmoid/tanh, plus normally initialized
Q/K/V/output projections and full-projection Q/K RMSNorm. This preserves a small
initial perturbation, not mathematically identical baseline outputs. New
projections are not zeroed. Audio/prompt padding, complementary visual masking,
and CFG video dropout are enforced on the local branch in training and inference.
Queries outside the generated audio region receive zero local residual, as do
examples with no valid visual keys.

The strict S2c 70k EMA migration still loads exactly **303** parent tensors.
The speaker model now has **770** state entries, including **66** additional
adapter tensors; **467** entries are newly initialized. Unexpected adapter
keys, incorrect shapes, and altered initial gates are rejected. Resume and
inference continue to use strict state-dict loading.

The new config is
`src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_ctc003_warmup.yaml`.
It inherits LR=5e-5, CTC 0 through 10k then linear to 0.03 at 30k, seed 666,
the 200-epoch LR schedule and stop at 200k updates. Its checkpoint directory is
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_local_visual_ctc003_warmup10k30k_40hz_CelebVDub_char`.

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_audio_local_visual_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py \
  --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_ctc003_warmup

mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_4x4090.sh \
  > logs/train_speaker_local_visual.log 2>&1 < /dev/null &
TENSORBOARD_LOGDIR="$PWD/runs/AlignDiT_MMDiT_c2_svae_speaker_local_visual_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char" \
  TENSORBOARD_PORT=6006 bash scripts/start_speaker_tensorboard.sh
```

TensorBoard also records `local_visual/layer_{12..17}/gate_mean`, `gate_absmax`,
and `gate_grad_norm`. Use the generic
`src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py` entry with this new
YAML as `--config` and the matching checkpoint as `--checkpoint`; the older
speaker evaluation launchers select the baseline config and checkpoint path.
Any AVSync gain must be measured after training with the same S1 evaluation
protocol and checkpoint update as the control; implementation checks alone
do not establish an improvement.
