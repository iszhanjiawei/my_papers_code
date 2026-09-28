# Semantic-VAE speaker model with cached AV-HuBERT InfoNCE

This independent experiment was copied from
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding` at source snapshot
`f7e135e`. Neither that baseline nor the SynchFormer experiment is modified.
The baseline source, configuration and utility files were copied; runtime
artifacts were excluded. There is no SynchFormer conditioning branch here.

## Supervision

The generator still predicts flow in normalized 64-D Semantic-VAE latents at
40 Hz. It retains the existing video conditions, CAM++ speaker conditions and
CTC schedule. The added training target is the final contextual AV-HuBERT
representation extracted from **original GT waveforms**, before any VAE
encoding or reconstruction. No teacher network or VAE decoder runs in training.

The student tap is the raw text cross-attention output in zero-based block 11
(the twelfth and final multimodal block), before its gate and residual addition.
A trainable linear projection maps 768 channels to 1024. This follows the
CoSyncDiT choice of representation; it is distinct from aligning the complete
post-residual audio hidden state in the earlier dual-role experiment.

Student positions are mapped from 40 Hz to the fixed native 25 Hz teacher grid.
The centered position for teacher index `j` is `(j + 0.5) * 40 / 25 - 0.5` in
student coordinates. This is a feature-grid convention, not a claim that a
contextual AV-HuBERT token represents an isolated 40 ms waveform segment. No
alignment uses the padded batch length or stretches a shortened valid prefix.
Only positions fully supported by valid student generation positions contribute
as anchors. The teacher's final padded filterbank/stack groups are excluded.

The loss uses L2-normalized projected student and detached teacher features:

```
positive: teacher frame j for student anchor j in the same utterance
negatives: valid teacher frames k of that utterance with abs(k - j) >= 5
temperature: 0.07
direction: student -> teacher
loss: mean negative log probability of the positive over valid anchors
```

The four neighboring frames on either side are excluded from negatives; five
teacher frames correspond to 200 ms. There are no cross-utterance or cross-rank
negatives. Repeated phonemes and silence may still produce false negatives;
this temporal exclusion does not remove that limitation. Loss computation uses
FP32. Each rank averages its valid anchors, and DDP averages rank gradients.

InfoNCE is enabled only on batches retaining both text and video conditions.
Classifier-free dropout, no valid anchor or no temporal negatives produces a
graph-connected zero so the projection remains safe under DDP. The projection
is saved in both online and EMA checkpoints; normal generation does not request
the auxiliary representation or require an audio teacher.

## Existing immutable teacher cache

The cache root follows the usual `ROOT_PREFIX` convention:

```
/zjw524/projects/data/CelebVDub/avhubert_audio_teacher_cache/
  aa9876bf51d0af280b87c575d53b8c0e408f1e3e677e74d341d45673f6b2a770/
```

Metadata binds `large_vox_iter5.pt`, audio-only final contextual output,
16 kHz PCM, 26-bin logfbank, stack order 4 and float16 `[T,1024]` arrays. Input
per-frame normalization recorded in that metadata is not output L2 normalization.
The original producer is the independent
`AlignDiT_mmdit_d1_hunyuan_dual_ca_allrope_avhubert_dual_role` experiment.

The reader checks the pinned metadata identity, current source-waveform resolved
path/size/mtime, embedded audio and teacher identities, shape, dtype and finite
values. It never creates, repairs or rewrites teacher features. Missing or stale
entries raise an error with the offending path.

Before implementation, a read-only audit matched all **79,613 / 79,613** current
training waveform identities to cache filenames, with no missing or extra keys.
Feature contents were validated for 32 deterministic samples; every consumed
feature is checked again by the loader. This is not a claim of an exhaustive
content audit. The 213 test items have no teacher entries in this namespace;
test-time generation does not need them.

## Training policy

| Setting | Value |
|---|---|
| Initialization | Same pinned S2c 70k EMA parent as the speaker baseline |
| Optimizer/update counter | New independent run |
| Hardware/precision | 4 RTX 4090 GPUs, bf16 |
| Batch | 3,600 latent frames per GPU, at most 32 samples |
| Peak LR / LR warmup | `5e-5` / 20,000 updates |
| LR horizon | Original 200-epoch schedule |
| CTC | 0 through update 10k, linear to 0.03 at 30k |
| InfoNCE | `0.05 * min(completed_updates / 10000, 1)`; first forward uses zero |
| Stop | 200,000 optimizer updates |
| Checkpoints | Numbered every 50k; `model_last.pt` every 5k |
| Seed | 666; training RNG 666 + rank |

Temperature, temporal exclusion and auxiliary weight are starting experimental
settings, not validated optima. An AVSync improvement is a hypothesis until the
generated-audio evaluation is complete. Preserve the baseline's 213-item,
same-clip-reference Setting 1 protocol for comparisons.

## Validation and launch

From this experiment directory, using the environment selected by `env.sh`:

```bash
source env.sh
export PYTHONPATH=src
PYTHON_BIN="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
"$PYTHON_BIN" scripts/test_audio_teacher_cache.py
"$PYTHON_BIN" scripts/test_avhubert_infonce.py
"$PYTHON_BIN" src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
"$PYTHON_BIN" scripts/audit_audio_teacher_cache.py \
  --output logs/audio_teacher_cache_full_coverage_audit.json
"$PYTHON_BIN" scripts/audit_audio_teacher_cache.py --sample-count 32 --check-features \
  --output logs/audio_teacher_cache_sample32_audit.json
```

The initial validation passed 10 cache contracts, 9 InfoNCE/model mechanism
tests, and the inherited speaker smoke checks. Real dataset/collation checks
also confirmed that enabling teacher reads leaves latent and video values
unchanged. A 947-frame latent example has 592 stored teacher frames and 591
fully supported teacher frames; the stored timeline is retained when masking
that final padded frame.

The new entry uses
`finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_avhubert_infonce.yaml`.
It authenticates the S2c parent, allows exactly the two new projection tensors
in addition to the speaker baseline, and saves a training contract with model,
data, teacher and optimization settings. Resume rejects a different contract.
Changing only the run limit, worker count or checkpoint/log locations does not
change the optimization identity.

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_avhubert_infonce_4x4090.sh \
  > logs/train_avhubert_infonce.log 2>&1 < /dev/null &
bash scripts/start_avhubert_infonce_tensorboard.sh
```

The launch script checks GPU availability and the rendezvous port, then uses
explicit four-process multi-GPU Accelerate. Runtime artifacts are excluded from
Git. Smoke runs must override `model.name` and `ckpts.save_dir` so they cannot
be resumed accidentally by the full experiment.

Default run name:

```
AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_avhubert_infonce_l12_w005_t007_gap5_semantic_vae_40hz_CelebVDub_char
```

Its TensorBoard logdir is `runs/<run-name>` within this snapshot. Rank 0 records
total, flow, CTC, raw/weighted InfoNCE, weights, valid anchors, negative counts,
positive/negative similarity, retrieval accuracy and projection gradient norms.
Losses are rank-0 batch diagnostics, not global validation metrics. Zero
InfoNCE on a dropped-condition batch is expected. The TensorBoard launcher
defaults to port 6006 and supports `TENSORBOARD_PORT` / `TENSORBOARD_LOGDIR`.

## Inference and evaluation

The independent launchers resolve this experiment's checkpoint/config paths:

```bash
bash src/aligndit/run/eval/infer_celebvdub_s1_svae_direct_speaker_avhubert_infonce.sh
bash src/aligndit/run/eval/eval_celebvdub_s1_svae_direct_speaker_avhubert_infonce.sh
```

They retain Setting 1, seed 0, EMA, 32 NFE, Euler/EPSS and text/video CFG 5/2.
`CKPT_STEP`, `EVAL_GPU`, `CFG_TEXT`, `CFG_VIDEO`, `NFE` and `OUTPUT_DIR` are
overridable. Inference checks the new checkpoint schema, model architecture,
normalization/vocabulary identities and teacher provenance, without opening the
teacher feature cache or loading its encoder. AVSync gains must be measured
after training; no metric improvement is asserted by these implementation tests.
