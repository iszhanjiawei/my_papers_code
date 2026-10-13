# Isolated Semantic-VAE C2 + speaker + adaptive temporal band

This is a source-code copy of `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding`.
The original project is unchanged. Source files are real copies, not shared imports
or symlinks. Training products were not copied. The existing dataset and frozen
speaker/codec caches are reused read-only.

## Intervention

Only audio-query/video-key attention logits in the first 12 joint blocks receive
the Gaussian temporal bias

`B[i,j] = -(t_video[j] - t_audio[i] - delta[i])**2 / (2*sigma[i]**2)`.

A single `LayerNorm(no affine) -> Linear(768,64) -> SiLU -> Linear(64,2)`
predictor reads the **branch-specific video embedding before joint attention**.
It is shared across heads and MM layers and has 49,346 trainable parameters.
It does not read noisy audio or speaker conditions. In CFG null/TTS branches it
reads the existing dropped-video embedding, never the full branch's video.

- Both model input rates are 40 Hz; the stored original video was already
  interpolated from 25 Hz by the unchanged cache pipeline.
- `delta = 0.100 * tanh(raw_delta)` seconds, positive meaning later video keys.
- `sigma = 0.025 + (0.250 - 0.025) * sigmoid(raw_sigma)` seconds.
- Initialize the final linear weight to zero, its bias to `[0, -log(2)]`:
  the starting prior has zero center offset and sigma 100 ms.
- Keep AA, VA and VV logits, all other branches, CTC taps/losses, and speaker
  injection unchanged. No new loss, teacher, encoder, hard mask or mass correction.
- The structural temporal bias is active during training even though the inherited
  trainer disables padding attention masks for memory reasons.
- Feature-disabled configurations retain the original parameter names and forward
  path. Feature-enabled initialization **does not** preserve baseline outputs:
  the 100-ms temporal prior is active from the first update.

This is an engineering test of the learnable Gaussian band, not a claim of a new
attention principle. In the retained joint softmax, nonpositive video bias also
reduces video attention mass at fixed logits. Learned width must not be interpreted
as calibrated alignment uncertainty, and bounded offsets do not guarantee monotonicity.

## Matched training protocol

Use the same SHA-pinned S2c 70k **EMA** checkpoint as the speaker baseline, with a
new optimizer and update counter; do not resume the completed speaker 200k run.
Keep seed 666 (rank RNG 666 + rank), LR 5e-5, LR warmup 20k, original 200-epoch
decay horizon, CTC 0 through 10k then linear to 0.03 at 30k, 4 GPUs, 3,600 latent
frames/GPU, bf16, gradient clipping 1.0 and stop at 200k. Save numbered checkpoints
every 50k and `model_last.pt` every 5k. All 79,613 training records are retained.

Strict migration still imports 303 S2c tensors and ignores the same ten HuBERT
projector tensors; the four temporal-band tensors are explicitly validated as new.
Initialization of existing tensors does not consume extra RNG from the predictor.

```bash
bash scripts/start_adaptive_band_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_adaptive_band_4x4090.sh \
  > logs/train_adaptive_band.log 2>&1 < /dev/null &
```

Config: `src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band.yaml`.
Checkpoint directory (under `ROOT_PREFIX`):

`/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_ctc003_warmup10k30k_40hz_CelebVDub_char`

TensorBoard serves this project's `runs/` on port 6008 by default. Formal run:

`AlignDiT_MMDiT_c2_svae_speaker_adaptive_band_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char`

Only rank 0 writes scalars. In addition to existing flow/CTC/speaker diagnostics,
record the valid-query mean/std/min/max of offset and sigma (milliseconds) and
the predictor's pre-clipping gradient norm. These are rank 0's local-batch
statistics, not cross-GPU averages. Smoke runs have their own model names
and checkpoint directories and are never resumed by the formal run.

## Verification and inference

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_adaptive_temporal_band.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
```

The copied `infer_celebvdub_s1_svae_direct_speaker_ctc003.sh` and corresponding
`eval_...sh` now explicitly select the **adaptive-band** config and checkpoint
directory. They retain the baseline Setting 1 same-clip reference protocol,
EMA, seed 0, CFG text/video 5/2 and 32 NFE. Inference loads all new weights strictly.
Older generic launchers remain for upstream compatibility; use the entries above
for this experiment, not an old baseline YAML with its original output directory.

### Setting 1 target-only text ablation

Set `TEXT_CONDITIONING=target-only` to send the target transcript once instead of
the historical same-clip `reference transcript + two spaces + target transcript`.
Only the model text changes: the same GT-clip reference latent and CAM++, the
`P+T` timeline, video, target latent crop, decoder and sampling settings are kept.
The historical char-text duration floor is retained in both modes: one of the
213 clips has a 58-frame audio/video layout but historically samples 67 frames.
Those nine extra tail frames remain outside the unchanged target crop.
The default remains `reference-target`. Target-only default output directories
have a `_target_only` suffix and cannot overwrite the historical results.

```bash
CKPT_STEP=150000 EVAL_GPU=0 TEXT_CONDITIONING=target-only \
    bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
CKPT_STEP=200000 EVAL_GPU=1 TEXT_CONDITIONING=target-only \
    bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

For background runs, launch each command with `setsid env` and unbuffered Python,
as required by the repository's `AGENTS.md`. Each inference summary records the
selected text mode and each clip's actual model text and prompt/target lengths.

The completed 213-clip EMA 150k/200k comparison is recorded in
[SETTING1_TARGET_ONLY_EVALUATION.md](SETTING1_TARGET_ONLY_EVALUATION.md).
Both checkpoints deteriorated on all four metrics; retain historical text
concatenation for this snapshot's Setting 1 protocol.

### Raw full-sequence diagnostic

Set `TEXT_CONDITIONING=target-only OUTPUT_REGION=raw-full` to decode
`trajectory[-1]` in full, before the sampler's reference-prefix replacement.
There is no latent crop, waveform crop, or reference-audio concatenation. The
reference latent, CAM++, target video and sampling parameters remain conditions.
Raw normalized latents are saved alongside the complete WAVs for auditing.
The historical effective sampling length is retained, including the one
67-frame exception above. The standard output remains `OUTPUT_REGION=target`.

```bash
CKPT_STEP=150000 EVAL_GPU=0 TEXT_CONDITIONING=target-only OUTPUT_REGION=raw-full \
    bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
CKPT_STEP=200000 EVAL_GPU=1 TEXT_CONDITIONING=target-only OUTPUT_REGION=raw-full \
    bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_ctc003.sh
```

WER, SPKSIM and EMOSIM read the **complete** WAV against the original single
target transcript/GT clip. AVSync is a separately labeled **target-window
diagnostic**: the mouth video only describes the target region, so its metric
audio is read in memory starting at `P*400` samples for the original target
sample count. The extractor records each window in `audio_region_protocol.json`
and never rewrites the complete WAV. Full-sequence and original target-only
metrics have different output-duration protocols; their comparison is diagnostic.

### Pre-launch verification (2026-09-12)

- 19 adaptive-band CPU regression tests passed, including exact legacy-disabled
  outputs, packed/separate CFG, null-video independence and checkpointed gradients.
- Existing speaker-only regression suite passed unchanged.
- CUDA BF16 audio/video attention backward passed at batch 8 and lengths 400/437;
  the predictor receives finite nonzero gradients.
- Four-GPU, three-update real-data run at the full 3,600-frame/GPU batch passed;
  strict migration: 313 source / 708 target / 303 loaded / 10 ignored / 405 new.
- Reloaded its checkpoint strictly as EMA; online delta and sigma output weights
  receive finite nonzero gradients on real data with CTC both 0 and 0.03.
- One real 2-NFE Setting 1 sample passed the full EMA -> latent -> VAE -> WAV path.
  This is a functional smoke test, not a quality evaluation.

Runtime evidence is stored under this snapshot's ignored `logs/` directory;
none of the smoke checkpoints/events or shared caches are committed to Git.
