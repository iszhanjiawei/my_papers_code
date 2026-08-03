# C2 Semantic-VAE integration

This directory is an isolated experiment snapshot copied from
`AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` at repository commit
`cc7c5a1`.  The original C0-C3 and D0-D2 mel experiments remain unchanged.

## Fixed experiment semantics

- Downstream architecture: C2, with 12 MM-DiT + text cross-attention blocks
  followed by 6 audio-only DiT blocks.
- Audio representation: Semantic-VAE 1000k posterior sample, 64 channels,
  40 Hz, 16 kHz waveform sample rate, and a 400-sample codec hop.
- Audio pretraining teacher: `facebook/hubert-large-ll60k`, final hidden
  state, 1024 channels.  Each utterance is interpolated from its native
  approximately 50 Hz length to the exact Semantic-VAE latent length.
- Cached posterior samples use a stable per-utterance seed derived from the
  utterance key and base seed 666.  They must not depend on GPU rank, world
  size, traversal order, or resume position.
- The authoritative first cache stores raw FP32 latent values.  Train-only
  per-channel statistics are recorded separately; normalization is a
  configurable later ablation and never overwrites the raw cache.

## External resources on the current `/home` layout

```text
/home/zjw524/datasets/LibriSpeech
/home/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE
/home/zjw524/projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k
/home/zjw524/projects/alignDiT_idea6/hubert-large-ll60k
```

Runtime code must keep using the repository's `ROOT_PREFIX` convention rather
than baking `/home` into portable configuration or launch scripts.

## Implementation order

1. Build a deterministic LibriSpeech inventory manifest.
2. Extract and validate fixed Semantic-VAE latent targets.
3. Extract HuBERT features and interpolate each utterance to its exact latent
   length.
4. Add the 64-channel latent dataset, CFM, time-preserving representation
   projector, exact-update trainer, and scratch pretraining entrypoint.
5. Transfer the 500k EMA audio path into C2 and fine-tune on CelebVDub.

Generated manifests, features, decoded audio, logs, and checkpoints live
outside this Git repository and must never be committed.
