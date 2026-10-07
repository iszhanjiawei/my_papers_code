# Semantic-VAE 6 MM / 12 audio: Flowley progressive visual window

This isolated project copies all 242 tracked source files from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avsync_local_visual_mm6_audio12_text_gate`
at commit `877c96aaea0e0e93425eb8df1623bdf9a2aca68b`. The new directory adds
`_progressive_window`. The source snapshot is unchanged; logs, TensorBoard events
and training checkpoints are not copied. Data and the pinned audio parent remain
external to each independent project.

## Exact experimental change

The new config selects `local_visual_window_schedule: flowley_progressive`.
All other resolved training settings are inherited from the text-gated source,
except for the experiment name and checkpoint directory. The backbone default
remains `fixed` for compatibility with the copied source configurations.

Flowley does not progressively shorten the geometric window radius. Its PSCA
keeps the core radius omega and fading width delta fixed, and multiplies only
the fade-zone weights by a depth-dependent coefficient:

```text
N = depth - n_text_layers = 12
l = block_index - n_text_layers = 0, ..., 11
beta_l = 1 - l / (N - 1)       (beta = 1 when N = 1)
fade_scale_l = base_fade_scale * beta_l
```

This snapshot retains omega=0, a 0.5-second outer radius, cosine decay and the
8-Hz reference coordinates. At the actual 40-Hz video rate this is 20 tokens on
either side. Every layer preserves a center weight of 1; progressively deeper
layers reduce the surrounding weights. The exact 12-layer schedule is:

| Audio-tail layer (1-based) | Backbone block (0-based) | beta / actual fade scale |
| --- | --- | --- |
| 1 | 6 | 1.000000 |
| 2 | 7 | 0.909091 |
| 3 | 8 | 0.818182 |
| 4 | 9 | 0.727273 |
| 5 | 10 | 0.636364 |
| 6 | 11 | 0.545455 |
| 7 | 12 | 0.454545 |
| 8 | 13 | 0.363636 |
| 9 | 14 | 0.272727 |
| 10 | 15 | 0.181818 |
| 11 | 16 | 0.090909 |
| 12 | 17 | 0.000000 |

For a distance d measured in reference-frame units, R=0.5*8=4:

```text
M_l(d) = 1                                      if d <= omega
         beta_l * 0.5 * (1+cos(pi*(d-omega)/(R-omega)))  if omega < d <= R
         0                                      if d > R
attention_bias_l = log(M_l + 1e-6)
```

The base fade scale is 1.0. The final layer has beta=0, so every non-center key
has bias log(1e-6), approximately -13.8155. It remains a soft suppression rather
than an exact hard cutoff; padding alone uses negative infinity. The code keeps
Flowley's `round` center alignment and uses configured frame rates rather than
padded sequence-length ratios. Prompt offsets and per-sample valid lengths keep
the previous behavior.

The schedule is deterministic model configuration, not a learned gate or a
training-step schedule. It is identical during training, inference and CFG.
It adds no parameter tensors or persistent buffers. State keys/shapes remain
identical to the 824-entry text-gated source model.

## Preserved architecture and pretraining

The first six blocks use MM-DiT joint attention and existing text CA. In all
12 audio-tail blocks, audio self-attention is followed by parallel text/visual
cross-attention and the original FFN. Both CA branches query the same post-SA
audio features. Text keys/values use the existing 512-D text context; visual
keys/values use the original cached 1024-D AV-HuBERT features.

Text and visual residuals retain separate learnable 768-channel gates, both
initialized to 1e-5. This static depth schedule only changes the visual
attention bias. It does not alter the gates, text branch, CAM++ speaker
conditioning in blocks 12..17, or CTC taps [6, 12]. Padding, generation masks,
modality dropout, packed CFG and activation checkpointing retain their behavior.

Start a new experiment from the same pinned S2c 70k EMA audio parent with a fresh
optimizer and update counter. Strict migration loads 303 audio tensors exactly,
ignores the same 10 parent projector tensors, and validates 521 new entries
including 132 visual and 132 text adapter entries. This run does not resume the
fixed-window experiment checkpoint.

Inherited training settings: 64-D / 40-Hz Semantic-VAE, 79,613 training records,
seed 666, bf16, four GPUs, 3,600 audio frames per GPU, LR=5e-5, 20k LR warmup,
200-epoch LR schedule, CTC disabled through 10k and linearly increased to 0.03
at 30k, and stop at 200k updates. Numbered checkpoints are saved every 50k;
`model_last.pt` is saved every 5k.

## Run from this project

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_progressive_visual_window.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_audio_local_visual_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_audio_tail_text_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py \
  --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_text_gate_progressive_window_ctc003_warmup

mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_text_gate_progressive_window_4x4090.sh \
  > logs/train_progressive_window.log 2>&1 < /dev/null &
TENSORBOARD_LOGDIR="$PWD/runs/AlignDiT_MMDiT_c2_svae_speaker_mm6_audio12_text_gate_flowley_progressive_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char" \
  TENSORBOARD_PORT=6006 bash scripts/start_speaker_tensorboard.sh
```

The independent checkpoint directory is
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_mm6_audio12_text_gate_flowley_progressive_ctc003_warmup10k30k_40hz_CelebVDub_char`.
The training contract records every actual layer's radius, core and fade scale.
Only rank 0 writes TensorBoard. Loss, CTC and gradient diagnostics plus all 12
text and 12 visual gates retain the source run's tags.

For inference, pass this project's new `...text_gate_progressive_window_ctc003_warmup.yaml`
to `src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py --config`, together
with this experiment's matching checkpoint. The YAML is essential: a state dict
alone does not encode the nonpersistent window schedule. Resume contracts check
the saved configuration; inherited configurations keep their fixed schedule.

## Flowley source reference

The user-provided local source is `../../papers_codes/Flowley`:

- `flowley/model/model.py`, `initialize_attention_mask`: beta decreases over
  single-stream blocks only, with beta=1 for a single block.
- `flowley/model/modules/layers/attention.py`,
  `compute_audio_visual_cross_attn_mask`: cosine fade, rounded time mapping and
  additive log(weight+1e-6).
- `configs/train.yaml`: omega=0, fade_range=4, cosine and video_fps=8.

The local paper's section 3.2 gives the same depth-dependent beta. The paper
writes a floor center mapping while the official code uses round; this project
follows the implementation as requested. Its pretrained speech backbone, dual
near-zero gates and fixed speaker/CTC protocol are retained, so this remains an
AlignDiT ablation based on Flowley PSCA rather than an exact Flowley reproduction.
