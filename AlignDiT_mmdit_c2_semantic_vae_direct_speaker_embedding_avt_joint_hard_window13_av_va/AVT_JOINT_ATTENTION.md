> Historical source-experiment record. For the current thirteen-frame run,
> see [AVT_HARD_WINDOW13_AV_VA.md](AVT_HARD_WINDOW13_AV_VA.md).

# Semantic-VAE speaker C2 with audio/video/text joint attention

This independent copy of `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding`
changes the first **12 MM-DiT blocks** from audio/video joint attention followed
by audio-to-text cross-attention to audio/video/text joint attention, with a
separate stream for each modality. The following **6 audio-only DiT blocks**
are retained unchanged. This experiment does not include adaptive temporal bands.

The attention topology follows `AlignDiT_mmdit_mmaudio_avt_joint_audio_pretrained`.
Its depth, data, audio representation and pretraining recipe are not imported.
Each modality has independent Q/K/V and an evolving residual state. All three
share one softmax over the concatenated `[audio, video, text]` keys. The source
audio/video AdaLN, attention projections and FFNs are retained; the text stream
adds a 512-to-768 projection, AdaLN, attention and FFN. There is no separate text
cross-attention in the selected model. Text receives no joint-level RoPE;
audio/video RoPE and the source padding policy are retained, with text padding
always excluded from the shared keys. The final MM block keeps the source's
full video sublayers, rather than introducing the reference's `pre_only` variant.
The source speaker experiment still supplies the Semantic-VAE codec,
64-channel/40-Hz cache, 192-dimensional CAM++ speaker condition on blocks 12–17,
CTC taps at layers 6/12 and unchanged flow/CTC training policy.
The full model has 376,563,842 parameters, versus 321,849,218 in the source.
The additional text stream changes parameter count; width and layer counts
remain unchanged.

The same SHA-pinned S2c 70k EMA initializes compatible audio weights, with a
new optimizer and update counter. The MM text stream requires new parameters.
Keep seed 666, 4 GPUs, 3,600 latent frames/GPU, bf16, LR 5e-5, LR warmup 20k,
CTC 0 through 10k then linear to 0.03 at 30k, and stop at 200k updates.

## Run this snapshot

The existing filenames are retained, but the speaker YAML sets
`model.arch.text_joint_attention: True` and uses isolated experiment names.
Run from this project directory:

```bash
bash scripts/start_speaker_tensorboard.sh
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 \
  bash src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_4x4090.sh \
  > logs/train_speaker_avt_joint.log 2>&1 < /dev/null &
```

- Training config: `src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_ctc003_warmup.yaml`.
- Checkpoints: `${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_40hz_CelebVDub_char_avt_joint`.
- TensorBoard run: `runs/AlignDiT_MMDiT_qknorm_ca_c2_semantic_vae_direct_speaker_ctc003_warmup10k30k_avt_joint_semantic_vae_40hz_CelebVDub_char`.
- Default ports: 29634 for DDP, 6018 for TensorBoard; override `TRAIN_PORT` or `TENSORBOARD_PORT` if needed.
- TensorBoard serves this copy's `runs/`; the original experiment is untouched.

The matching `infer_celebvdub_s1_svae_direct_speaker_ctc003.sh` and
`eval_celebvdub_s1_svae_direct_speaker_ctc003.sh` under `src/aligndit/run/eval/`
select this copy's config and the new checkpoint directory. Older upstream
configs and historical documentation remain available; use the entries above
for this experiment.

## Verification (2026-10-12)

```bash
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_avt_joint.py --cuda
PYTHONPATH=src /zjw524/ENTER/envs/aligndit/bin/python -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker.py
```

- Nine CPU AVT regressions and the CUDA BF16 checkpointed backward check pass.
  They verify shared-softmax numerical equivalence, bidirectional interaction,
  text padding, text state propagation, CFG packing, checkpointed gradients,
  speaker conditioning, CTC taps, and the unchanged 12+6 staging/configuration.
- The inherited speaker regression passes on the unchanged legacy default.
  The AVT regression separately verifies speaker conditioning in the new topology.
- Strict migration loads all 303 compatible S2c EMA tensors exactly, ignores
  the same ten source projector tensors, and validates all 499 new tensors
  against an explicit whitelist (802 target tensors in total).
- A real-data 4-GPU run at the unchanged 3,600 frames/GPU and BF16 completed
  three updates with finite loss, saved `model_last.pt`, and recorded TensorBoard
  loss/flow/CTC-weight/speaker/gradient scalars. Activation checkpointing remains
  disabled in the production config, matching the source.
- The smoke run uses the model-name suffix `_smoke3_20261012` and checkpoint
  directory suffix `_smoke3_20261012`, with `optim.run_until_update=3`.
  Formal training uses the original 200k stop and starts independently from S2c.

Runtime logs and verification reports are under the ignored `logs/` directory;
event files and checkpoints are not versioned. Original-source SHA256 checks
confirmed all 234 copied files remained unchanged in the source project.
