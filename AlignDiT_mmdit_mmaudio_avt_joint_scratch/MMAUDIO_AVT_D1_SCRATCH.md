# MMAudio AVT Joint D1 Scratch Contract

This snapshot is an isolated CelebVDub experiment derived from D1. It must not
modify or resume the completed C0-C3/D0-D2 experiments in
`AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio/`.

## Fixed experiment identity

- Dataset: CelebVDub only.
- Initialization: random/AdaLN-Zero scratch initialization. Do not load the
  LibriSpeech pure-audio AlignDiT checkpoint. Linear layers use MMAudio-style
  Xavier initialization before AdaLN and the final output are zeroed.
- Audio representation: 80-dimensional mel, 16 kHz, 100 Hz.
- Video representation: 1024-dimensional AV-HuBERT, native 25 Hz.
- Text representation: ordered character tokens, character positional
  encoding, four padding-safe ConvNeXt blocks, then a 512-to-768 joint-stream
  projection. GRN statistics never include padded characters.
- Backbone: 18 layers total.
  - Layers 0-5: Audio/Video/Text three-stream MMAudio-style Joint Attention.
  - Layer 5: Video/Text are `pre_only`; Audio is still updated.
  - Layers 6-17: Audio-only DiT refinement.
- Position encoding: Audio positions `i`; Video positions `4j`; no
  joint-level Text RoPE.
- CTC: Audio hidden states after zero-based layers `[5, 11]`, weight `0.1`.
- QK normalization: RMSNorm.
- RoPE heads: one head, retained from D1 for the topology ablation.
- Activation checkpointing: enabled so the D1 9000-frame/GPU budget fits
  24-GiB RTX 4090 training; it does not change the network function.
- Length contract: every valid mel length must equal four times its native-rate
  video-token length; training and inference fail fast instead of truncating.

This model has about 336M parameters versus about 250M for the original D1.
It is therefore a D1-width, MMAudio-block experiment rather than a
parameter-matched topology ablation. A parameter-matched follow-up should use
a separately named configuration. Likewise, `pe_attn_head: 1` deliberately
retains D1's RoPE setting; original MMAudio applies RoPE to all heads, so an
all-head variant must be reported as a separate ablation.

## Formal entrypoints

- Config:
  `src/aligndit/config/finetune_celebvdub_mmaudio_avt_joint_d1_scratch.yaml`
- Launcher:
  `src/aligndit/run/train/finetune_celebvdub_mmaudio_avt_joint_d1_scratch_4x4090.sh`
- Backbone:
  `src/aligndit/model/backbone/dit_vt_mmaudio.py`

The config must retain:

```yaml
ckpts:
  init_mode: scratch
  pretrained_path: null
```

The unique `model.name` and `save_dir` prevent the trainer's normal
same-experiment resume behavior from discovering an old D1 checkpoint. A later
restart of this experiment may resume its own `model_last.pt`; that is not
audio-pretrained initialization.

## Joint Attention invariants

Each joint block independently computes Q/K/V for Audio, Video, and Text, then
concatenates all three on the token dimension. The key mask order must be
`[audio_mask, video_mask, text_mask]`. Padding masks remain active in training.
The character stream keeps ordinal position features from its encoder but is
not assigned the Audio/Video physical-time RoPE coordinate system.

Do not restore a separate Text Cross-Attention branch in this experiment: that
would confound the comparison between D1 and the MMAudio-style shared-softmax
topology.

## Validation and launch

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_mmaudio_avt.py

CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=src \
  /zjw524/ENTER/envs/aligndit/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 \
  src/aligndit/script/misc/smoke_test_mmaudio_avt.py --ddp-full

bash src/aligndit/run/train/finetune_celebvdub_mmaudio_avt_joint_d1_scratch_4x4090.sh
```
