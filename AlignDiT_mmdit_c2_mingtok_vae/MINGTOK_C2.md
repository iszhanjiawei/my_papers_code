# C2 with MingTok 64D acoustic latents

This snapshot starts from the mel C2 experiment in
`AlignDiT_mmdit_base_qknorm_ca_solve_prompt_audio` and replaces its audio
representation with the released MingTok-Audio acoustic latent.  It is an
independent experiment: changes in this directory must not be copied back to
the completed mel C0-C3 snapshots.

The mel-domain scratch experiment introduced by commit `3eec50f` is a parallel
control: it keeps the original 80D/100 Hz mel representation and only removes
the audio-pretraining load.  This directory does not copy that mel experiment;
it independently combines the original C2 architecture with MingTok's raw
64D/50 Hz acoustic representation and CelebVDub scratch initialization.

## Experiment contract

| Item | Contract |
|---|---|
| AlignDiT initialization | Fresh random/AdaLN-Zero initialization on CelebVDub |
| AlignDiT parent checkpoint | None; LibriSpeech audio-pretraining weights are forbidden |
| Codec | Local MingTok-Audio checkpoint, acoustic branch only |
| Codec representation | Raw posterior sample, 64 channels, 50 Hz |
| Codec normalization | None (no channel mean/std, scalar scale, clamp, or latent LayerNorm) |
| Codec training | Frozen and external to AlignDiT; never stored in an AlignDiT checkpoint |
| Video representation | AV-HuBERT, 1024 channels, native 25 Hz |
| Audio/video ratio | 2 latent frames per video frame |
| C2 backbone | 18 blocks; blocks 0-11 MM-DiT with text CA, blocks 12-17 audio-only |
| Text CA | Global (`prompt_isolated_ca=false`) in the first 12 blocks |
| CTC taps | Zero-based blocks 6 and 12 |
| CTC frame rate | 50 Hz (`ctc_sampling_ratios: [1, 1]`) |
| Training data | All 79,613 CelebVDub training records; no CTC-based filtering |
| Cache selection | Exact `CelebVDub_char/raw.arrow` `audio_path` rows, not a directory scan |
| Experiment seed | 666 for model initialization, rank-specific training RNG, and dynamic-batch ordering |
| Latent sampling seed | 666, used to make each cached VAE posterior sample deterministic |
| Dynamic batch | 4,500 latent frames/GPU (about 90 seconds/GPU) |
| Optimizer | AdamW, one learning rate of 5e-5, 20k warmup, max grad norm 1.0 |
| Auxiliary loss | CTC weight 0 through update 10k, then linearly ramped to 0.01 at update 30k |
| EMA | beta 0.999 |

The local codec is pinned by these hashes:

```text
config.json       e65fa0aec76f058308f75a4f7f892d8bdb3a3a7d79116b2b163f36b00d118c4a
model.safetensors c36d876de086d13eb1cdcfb9d08e22c3d806cd7893d64fdaf7ea6d30b7d521cd
CelebVDub raw.arrow 99da14538f85eca3a039282d1cb5126f2a5598dd3c513422fe58b454af9437ef
```

## Why the time axes use a 2:1 ratio

MingTok uses a 320-sample hop at 16 kHz, hence 50 latent frames per second.
CelebVDub AV-HuBERT features are 25 Hz.  The dataset keeps video at its native
rate and deterministically trims or replicate-pads each cached latent to
exactly twice the video length.  No video interpolation is needed.

The original mel C2 CTC projector used strides `[2, 1]` to convert 100 Hz mel
states to about 50 Hz.  Applying the same projector to MingTok would reduce CTC
to 25 Hz.  This snapshot instead uses `[1, 1]`, so CTC remains at 50 Hz.

## Data flow

```text
CelebVDub wav (mono, 16 kHz)
  -> frozen MingTok encoder
  -> deterministic raw posterior sample [T, 64]
  -> per-utterance FP32 cache

cache [T, 64] + AV-HuBERT [V, 1024]
  -> enforce T = 2V
  -> CFM / C2 MMDiT
  -> generated raw MingTok latent [T, 64]
  -> frozen MingTok acoustic decoder
  -> 16 kHz waveform
```

The audio tree contains files outside the original C2 metadata selection, so
formal extraction is keyed by the 79,613 `raw.arrow` rows.  This keeps the
MingTok experiment's training examples identical to mel C2.

At startup, the dataset pins the `raw.arrow` SHA256 and requires the cache
contract to match all of: posterior `sample`, base seed 666, 79,613 items,
raw-arrow ordering, no normalization, and the released MingTok config/model
hashes.  A same-shaped posterior-mean or differently sampled cache is rejected.

The optimization path reads the latent cache and never instantiates the full
MingTok VAE or its 1280D semantic branch.  This scratch experiment uses
`log_samples=false`, so training does not load the frozen acoustic decoder.
No codec parameter enters the optimizer, EMA, or AlignDiT checkpoints.

The local Transformers 4.51 runtime prints a generic warning that sliding
window attention is not implemented for eager attention.  Its Qwen2 model
still constructs and applies the checkpoint's 32-frame 4D causal window mask;
the codec smoke test uses this eager path, which is also the path used for the
completed codec-ceiling measurement.

## Strict C2 boundary

Only changes required by the MingTok representation are allowed: 64D input and
output, 50 Hz latent loading and decoding, a 2:1 audio/video ratio, a 50 Hz CTC
projector, and the removal of the initial LibriSpeech AlignDiT load.  The C2
text path, loss, optimizer, scheduler, EMA, dropout, attention-mask behavior,
gradient clipping, and checkpoint/resume implementation remain unchanged.

In particular, this experiment does not add text-context LayerNorm, extra
gradient-norm instrumentation, optimizer-step non-finite fail-fast logic, a
changed CTC class mapping, or unrelated mask/checkpoint fixes.  The dedicated
entry calls the existing training loop directly instead of `finetune()`, so first launch starts
from the model's normal random/AdaLN-Zero initialization while an existing
checkpoint in this experiment's independent save directory resumes with the
original C2 behavior.

## Intended workflow

Run every command from this project directory with the existing AlignDiT
environment:

```bash
# 1. Verify the real local MingTok encoder and decoder on one GPU.
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src \
  /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_mingtok_codec.py

# 2. Build all 79,613 deterministic train latents and merge/validate them.
bash src/aligndit/run/misc/extract_mingtok_latents_celebvdub_4x4090.sh

# 3. Verify the 64D/50Hz dataset, CFM, CTC length and ODE contracts.
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_mingtok_c2.py

# 4. Start C2 training from CelebVDub (or resume this experiment only).
# Long runs must be detached from the controlling terminal.  Keep the PID and
# verify that SID equals PID and TTY is "?" before reporting a successful start.
run_id="$(date +%Y%m%d_%H%M%S)"
log="logs/train_celebvdub_mingtok_c2_ctc001_warmup10k30k_seed666_4x4090_${run_id}.log"
pid_file="${log%.log}.pid"
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/train_celebvdub_mingtok_c2_4x4090.sh \
  > "$log" 2>&1 < /dev/null &
train_pid=$!
printf '%s\n' "$train_pid" > "$pid_file"
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"

# 5. After a checkpoint exists, run four-GPU Setting-1 EMA inference.
bash src/aligndit/run/eval/infer_celebvdub_mingtok_s1_4x4090.sh \
  /absolute/path/to/model_STEP.pt
```

The full cache build and long training run are deliberate operations and are
not started merely by running a smoke test.
