# Semantic-VAE 6 MM / 12 audio: gated text and local visual conditioning

This independent snapshot copies all 238 tracked source files from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avsync_local_visual_mm6_audio12`
at commit `e5fadd4ac9911a27502439cbea9aab3f705b7be8`. Its project directory adds
`_text_gate`. The source snapshot is not edited; runtime logs, checkpoints and
TensorBoard files are not copied. Data and the pinned audio parent remain external.

## Experimental change

Total depth remains 18. All indices below are zero-based.

| Component | Visual-only source | This experiment |
| --- | --- | --- |
| MM-DiT joint-attention blocks | 0–5 | 0–5 |
| Existing text CA in MM blocks | 0–5 | 0–5 |
| Audio self-attention and FFN | all 18 blocks | all 18 blocks |
| Local visual CA | 6–17 | 6–17 |
| New gated text CA in audio blocks | absent | 6–17 |
| CAM++ speaker modulation | 12–17 | 12–17 |
| CTC taps | [6, 12] | [6, 12] |

The new `audio_tail_text_attention` flag adds text conditioning to the existing
`AudioVisualDiTBlock`; `n_text_layers=6` retains its original block-factory meaning.
It must not be changed to 18, which would construct a different block type and
remove the existing local visual attention. The new flag defaults to false so
inherited visual-only configurations retain their previous architecture.

Each new text adapter has its own query, key, value and output projections,
projected Q/K RMSNorm, and a learnable 768-channel residual gate initialized to
`1e-5`. Text keys/values use the existing 512-D encoded text, without another text
encoder. The visual branch retains its own independent gate. Near-zero gates
reduce initial residual perturbation; they do not freeze the pretrained backbone
or guarantee unchanged weights during training. Setting the text gates exactly
to zero restores the visual-only model when shared weights are identical.

The tail block order is audio self-attention, parallel text/visual
cross-attention, then the original FFN. Both cross-attentions read the same
post-self-attention audio states:

`x_next = x + g_text * CA_text(x, text) + g_visual * CA_visual(x, video)`.

The visual module already applies its gate internally. No extra 0.5 mixing
coefficient is added, so enabling a zero-gated text branch preserves the
existing visual residual exactly.

Text padding and invalid audio queries are masked. The C2 default
`prompt_isolated_ca=false` keeps text conditioning on all valid audio queries;
when enabled, the adapter limits text queries to the generated region. CFG text
dropout disables the new text residual, including output bias. Video dropout
does not disable text conditioning. Packed CFG and activation checkpointing use
the same branch masks as ordinary forward passes.

The local visual branch remains based on raw cached 1024-D AV-HuBERT features.
Its fixed Flowley-style cosine window has radius 0.5 s (20 tokens at 40 Hz),
with additive `log(weight + 1e-6)` bias and padding excluded by negative infinity.
Its separate OmniShow-style channel gate remains initialized to `1e-5`.

## Pretraining and training protocol

Initialization is the same pinned S2c 70k EMA audio parent, with a fresh optimizer
and update counter. This experiment does not resume the visual-only training
checkpoint. Strict migration retains the original 303 audio tensors and ignores
the same 10 parent projector tensors. It validates every new adapter key, shape
and fresh gate. The 692-entry visual-only model gains 132 text-adapter entries,
for 824 state entries (303 loaded, 521 new).

Inherited settings: 64-D / 40-Hz Semantic-VAE, 79,613 training records, CAM++
embeddings, seed 666, bf16, four GPUs, 3,600 audio frames per GPU, LR=5e-5,
20k LR warmup, 200-epoch LR schedule, CTC disabled through 10k and linearly
increased to 0.03 at 30k, and stop at 200k updates. Numbered checkpoints are
saved every 50k; `model_last.pt` is saved every 5k.

The fully composed training config differs from the source only in the two new
text-adapter options, experiment name and checkpoint output directory.

## Run from this project

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_audio_local_visual_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  scripts/test_audio_tail_text_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py \
  --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_text_gate_ctc003_warmup

mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_local_visual_text_gate_4x4090.sh \
  > logs/train_mm6_audio12_text_gate.log 2>&1 < /dev/null &
TENSORBOARD_LOGDIR="$PWD/runs/AlignDiT_MMDiT_c2_svae_speaker_mm6_audio12_local_visual_text_gate_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char" \
  TENSORBOARD_PORT=6006 bash scripts/start_speaker_tensorboard.sh
```

Checkpoints use the separate directory
`${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_mm6_audio12_local_visual_text_gate_ctc003_warmup10k30k_40hz_CelebVDub_char`.
Only rank 0 writes TensorBoard. In addition to loss, CTC and global gradients,
`tail_text/layer_6..17/` and `local_visual/layer_6..17/` each record gate mean,
absolute maximum and gradient norm. Resume restores model, EMA, optimizer and
scheduler; the training contract prevents reuse with mismatched configuration.

For inference, use `src/aligndit/script/eval/infer_celebvdub_semantic_vae_s1.py`
with the new `...local_visual_text_gate_ctc003_warmup.yaml` as `--config` and its
matching checkpoint. Keep the same S1 evaluation protocol for comparisons.

## Reference implementation decisions

The user-provided local reference repositories are `papers_codes/Flowley` and
`papers_codes/OmniShow` under the workspace root.

- [Flowley single-stream implementation](https://github.com/Fsoft-AIC/Flowley/blob/main/flowley/model/modules/single_modal_stream.py)
  computes its text/visual cross-attentions in parallel. Its learnable
  `cross_weight` starts at 0.5 and mixes the two branches; this is different from
  a near-zero residual gate.
- [Flowley model initialization](https://github.com/Fsoft-AIC/Flowley/blob/main/flowley/model/model.py)
  and its default training configuration construct the generation backbone from
  random initialization rather than loading a pretrained text-to-audio backbone.
  This does not mean the system has no pretrained components:
  [its model documentation](https://github.com/Fsoft-AIC/Flowley/blob/main/docs/MODELS.md)
  lists pretrained encoders, audio VAE/vocoder and released Flowley checkpoints.
- [OmniShow section 3.3](https://arxiv.org/html/2604.11804v1#S3.SS3)
  motivates a per-channel attention-residual gate initialized to 1e-5. Its local
  `gated_local_context_attention.py` implements that gate directly. This
  experiment applies the same gate form independently to text and visual
  residuals. It is an adaptation of these components, not an exact Flowley model.
