# Local AV + blocked VA + WavLM-Base+ REPA

This independent project starts from the tracked Step-1 source snapshot. It adds
the VA-blocking behavior from Step-3 and the WavLM REPA supervision from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus`.
All three reference projects remain unchanged. No logs or trained weights were
copied into this source snapshot.

## Experiment contract

| Attention region | Behavior in each of the first 12 MM-DiT blocks |
| --- | --- |
| Audio queries / audio keys (AA) | Global |
| Audio queries / video keys (AV) | Radius 2 on the common 40-Hz grid (+/-50 ms) |
| Video queries / audio keys (VA) | Blocked |
| Video queries / video keys (VV) | Global |

Audio queries retain one shared softmax over audio and permitted video keys.
With VA blocked, video queries attend directly to video keys. This is the same
attention topology as masking the entire VA quadrant; it is not a separate
normalization of AA and AV. Structural restrictions remain active in training
even when the inherited training path does not supply a padding mask.

There is no visual-specific gate, no local VV mask and no adaptive Gaussian
band. Existing AdaLN residual modulation, text conditioning, CAM++ tail-six-layer
speaker conditioning and the remaining six audio-only blocks are retained.
The video branch remains trainable through the audio prediction loss.

REPA matches the reference experiment:

- Teacher: frozen `microsoft/wavlm-base-plus`, revision
  `4c66d4806a428f2e922ccfa1a962776e232d487b`, layer 12, 768 dimensions.
- Reuse the complete 79,613-record GT utterance cache at
  `${ROOT_PREFIX}/zjw524/projects/data/CelebVDub/wavlm_base_plus_repa_final_fp16`.
  Validate teacher identity, manifest, coverage, shapes and finite values.
- Interpolate each valid 50-Hz teacher sequence to its own valid 40-Hz latent
  length before padding. No teacher network is loaded in training workers.
- Tap the audio output of zero-based block 9 (the 10th MM-DiT block); use the
  same `768 -> 2048 -> 2048 -> 768` SiLU projection head.
- Reduce `1 - cosine_similarity` over the same generation mask as the flow
  loss, excluding prompt and padding frames. Teacher targets are detached.
- Use `loss = diff_loss + 0.1 * repa_loss + ctc_lambda(update) * ctc_loss`.
  CTC stays zero through 10k updates and reaches 0.03 at 30k.
- REPA is a training objective. Sampling does not call the projection head or
  require teacher features, but strict EMA loading retains the head parameters.

Initialize from the same SHA-pinned S2c 70k EMA audio parent, with a fresh
optimizer. Do not resume a trained Step-1, Step-3 or adaptive-band REPA checkpoint.
Preserve seed 666, LR 5e-5, 20k LR warmup, bf16, 3,600 frames/GPU, at most
32 samples/GPU, the 200-epoch LR schedule, and the 200k-update stopping point.
Keep full checkpoints at 50k/100k/150k/200k and refresh the last checkpoint every
5k updates. The new REPA head is created after the inherited modules to preserve
their seeded initialization.

This combined run changes both VA connectivity and REPA relative to Step-1.
It cannot by itself measure the isolated contribution of closing VA; that would
require the otherwise identical no-REPA comparison.

## Validation

Completed on 2026-09-28 before starting the formal four-GPU run:

- Eight topology contract groups passed, including output/gradient equivalence
  to explicit four-quadrant masked attention, zero VA gradients, live local AV
  and global AA/VV, training with no padding mask, and rectangular A/V lengths
  in actual text-clamped CFM sampling and packed CFG.
- Eight combined REPA tests passed: identical inherited seeded weights,
  inference without the auxiliary head, generation-only loss, detached teachers,
  gradients through local AV, activation-checkpoint parity, and strict EMA reload.
- Existing generic REPA and speaker-conditioning contract suites passed.
- Hydra resolved the intended configuration; Python compilation, shell syntax
  and Git whitespace checks passed.
- The actual S2c parent passed SHA validation and exact tensor migration:
  313 source tensors, 710 target tensors, 303 loaded, 10 ignored, 407 new.
- Two real clips (126/98 latent frames; 156/121 teacher frames) passed BF16
  forward/backward with CTC weights 0 and 0.03. Total losses were 1.599816 and
  1.879645; REPA loss was 0.996341. Video, speaker and REPA gradients were finite
  and nonzero. This check performed no optimizer updates. Its 2.63-GiB peak
  describes the small validation batch, not the formal training memory demand.

Repeat the contracts from this project's root:

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_av_local_no_va.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_av_local_no_va_repa.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_semantic_vae_c2_repa.py
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
# Select an available GPU for this integration check; it is not a training run.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_av_local_no_va_repa_real_parent.py
```

## Entry points and outputs

Primary configuration:

```text
src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_av_local_no_va_repa_wavlm_base_plus.yaml
```

Run from this project directory:

```bash
mkdir -p logs
bash scripts/start_av_local_no_va_repa_tensorboard.sh
setsid env PYTHONUNBUFFERED=1 bash src/aligndit/run/train/finetune_celebvdub_svae_speaker_av_local_no_va_repa_wavlm_base_plus_4x4090.sh > logs/train_av_local_no_va_repa.log 2>&1 < /dev/null &
```

The launcher uses physical GPUs 0/1/2/3 and distributed port 29635
(`TRAIN_PORT` override). TensorBoard defaults to port 6015
(`TENSORBOARD_PORT` override) and serves this project's `runs/` directory.
Record the actual PID, run logdir and reachable address at launch. Only rank 0
writes events, including total/flow/CTC/REPA losses and REPA projector gradients.
Runtime logs, events, checkpoints and validation reports are ignored by Git.

Checkpoint directory:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_av_local_no_va_repa_wavlm_base_plus_ctc003_warmup10k30k_40hz_CelebVDub_char
```

Inference and full four-metric evaluation are prepared for checkpoints from
this run; these commands are alternatives because the full evaluation includes
inference and refuses to overwrite existing output:

```bash
CKPT_STEP=150000 EVAL_GPU=0 bash src/aligndit/run/eval/infer_celebvdub_s1_svae_speaker_av_local_no_va_repa_wavlm_base_plus.sh
CKPT_STEP=200000 EVAL_GPU=0 bash src/aligndit/run/eval/eval_celebvdub_s1_svae_speaker_av_local_no_va_repa_wavlm_base_plus.sh
```

The protocol stays CelebV-Dub Setting 1, 213 clips, same-clip GT reference,
EMA, seed 0, Euler/EPSS, 32 NFE, CFG text/video 5/2 and true duration.
The legacy speaker training/evaluation/TensorBoard shell entries redirect to
this combined experiment. Other inherited upstream experiment scripts and
documents are historical references, not launch entries for this run.
