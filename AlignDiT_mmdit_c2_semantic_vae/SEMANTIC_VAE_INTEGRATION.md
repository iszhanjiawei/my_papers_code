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

## Frozen LibriSpeech inventory

The authoritative manifest is stored outside Git at:

```text
/home/zjw524/projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/manifests
```

It contains 281,241 training records and 5,551 development records. All three
training subsets are retained. Sixteen development utterances longer than 30
seconds are recorded in `rejected.jsonl`. The authoritative inventory SHA256
is:

```text
65c1332f9852bb84ddba8cfef8359cf5f2c7195a593d4e24087eb6c60d1dabe5
```

## Semantic-VAE latent extraction

The extractor uses only the exact 145 checkpoint tensors needed by
`encoder`, `pre_block`, `fc_mu`, and `fc_var`. It does not move the BigVGAN
decoder or distillation projector to each GPU. Every output is FP32 with shape
`[ceil(num_samples / 400), 64]`.

Correctness safeguards include:

- exact `records[rank::world_size]` ownership without sampler padding;
- a stable per-utterance CUDA generator seed from the manifest;
- a discarded 400-sample CUDA warm-up before any real posterior statistics;
- a built-in golden test whose raw latent SHA256 is
  `e3de5ff47682f97e063c6aaeaee9cec195ebdb34e1bce964c4a10d2912114f3f`;
- same-directory NPY temp files, file `fsync`, and hard-link no-clobber
  publication on NFS;
- immutable per-attempt/rank progress logs, a consolidated manifest-order
  index, and a final completion marker;
- CPU-only read validation and explicit offline repair that quarantines rather
  than deletes damaged or orphaned files.

The golden utterance is `train-clean-100/103/1240/103-1240-0015`. Its stored
NPY SHA256 must be:

```text
95774c4fb6dce29c18740bc5d1bc6630f4ba9a8dbde50460f7eb8c527d431b84
```

Use the isolated Semantic-VAE environment. It intentionally differs from the
main AlignDiT environment, which does not contain `audiotools`:

```bash
cd /home/zjw524/projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_semantic_vae

setsid env ROOT_PREFIX=/home NPROC_PER_NODE=8 PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/misc/extract_librispeech_svae_latents.sh \
  > logs/extract_librispeech_svae_latents.log 2>&1 &
```

The launcher calls the environment's Python module directly:

```text
/home/zjw524/ENTER/venvs/semantic-vae/bin/python -m torch.distributed.run
```

Do not substitute the main environment's `torchrun`. A writing launch creates
`state/latents/WRITE_ACTIVE.json`. If a process was killed, first verify that
no extractor is alive, then resume with a new attempt ID and explicitly name
the stale attempt:

```bash
ROOT_PREFIX=/home SVAECACHE_ATTEMPT_ID=<new-id> \
  bash src/aligndit/run/misc/extract_librispeech_svae_latents.sh \
  --acknowledge-stale-write-attempt <old-id>
```

Read-only validation does not require CUDA and never writes the cache:

```bash
ROOT_PREFIX=/home CUDA_VISIBLE_DEVICES= PYTHONPATH=src \
  /home/zjw524/ENTER/venvs/semantic-vae/bin/python -u \
  src/aligndit/script/misc/extract_librispeech_svae_latents.py \
  --validate-only
```

`--repair` is deliberately restricted to a full-manifest, single-GPU,
offline run. It invalidates the completion marker while working, quarantines
corrupt outputs, stale temp files and unexpected files, regenerates exact
posterior samples, and republishes completion only after a full audit.
