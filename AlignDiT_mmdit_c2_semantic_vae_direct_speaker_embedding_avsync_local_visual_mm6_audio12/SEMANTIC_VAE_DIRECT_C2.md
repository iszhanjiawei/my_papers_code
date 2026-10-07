# Semantic-VAE Direct-C2 experiment contract

This snapshot is copied from `AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` and keeps the original C2
architecture and optimization policy. It tests one controlled change: replacing the 80D/100 Hz mel target with the
fixed 64D/40 Hz Semantic-VAE latent.

## Required representation changes

- Read fixed-posterior 64D/40 Hz latent caches and normalize them with the pinned LibriSpeech-train mean/std.
- Read the cached AV-HuBERT video features interpolated from 25 Hz to the exact latent length at 40 Hz.
- Use `audio_video_ratio=1` and unscaled video RoPE positions.
- Keep both CTC taps at blocks `[6, 12]`, but use projector strides `[1, 1]` so CTC remains at 40 Hz.
- Use 3,600 latent frames/GPU, equal to the original C2 exposure of 9,000 mel frames/GPU (90 seconds/GPU).
- Strictly migrate the 303 compatible S2c-70k EMA state tensors; ignore only the 10 S2c HuBERT projector tensors;
  leave the 400 original-C2 multimodal/text/video/CTC tensors at their C2 initialization.

The complete 79,613-record C2 train split is retained. The 105 records that cannot satisfy CTC at 40 Hz still
contribute diffusion loss; the unchanged C2 `zero_infinity=True` behavior makes only their CTC term zero.

## Deliberately unchanged from C2

- 18 blocks: 12 MM-DiT blocks followed by six native text-free audio DiT blocks.
- Global text CA in the first 12 blocks, no prompt isolation, and RMS QK-Norm.
- One-stage training of every trainable parameter with one AdamW group at `5e-5`.
- 20k warmup, fixed CTC weight `0.1` from update 1, gradient clipping at 1.0, and 200 epochs.
- EMA configuration contains only `beta=0.999`, preserving ema-pytorch's C2 defaults.
- Original C2 text padding/mask behavior and video-null buffer behavior.

There is no staged freeze/unfreeze, grouped learning rate, CTC ramp, extra text LayerNorm, raw-RMS abort threshold,
or 200k scheduler override in this experiment.

## Entry points

```text
Config:   src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct.yaml
Python:   src/aligndit/script/train/finetune_semantic_vae_c2_direct.py
Launcher: src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_4x4090.sh
Smoke:    src/aligndit/script/misc/smoke_test_semantic_vae_c2_direct.py
Output:   /zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_c2_40hz_CelebVDub_char
```

The experiment must start from the S2c `model_70000.pt` EMA. Checkpoints from the earlier Semantic-VAE S3 variants
are not compatible parents for this controlled comparison because they contain additional training-policy changes.
