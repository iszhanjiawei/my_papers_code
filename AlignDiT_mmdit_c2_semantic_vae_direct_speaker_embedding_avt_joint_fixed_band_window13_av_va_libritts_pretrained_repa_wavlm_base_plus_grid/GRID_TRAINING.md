# GRID fixed-13 AVT + LibriTTS + WavLM-Base+ REPA

This is an isolated copy of
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_fixed_band_window13_av_va_libritts_pretrained_grid`
at Git `d8dc16758373479064241ff6bd1d02e94605172d`. Its 286 tracked files were copied
byte-for-byte and recorded in the standalone snapshot commit
`d9299ae1dc89030d4f0555c2ff4035093c954bf5`. Runtime logs, events, generated samples
and checkpoints were not copied. The source experiment remains unchanged.

The user requested a controlled REPA addition, fresh LibriTTS initialization,
and training through **150,000 GRID updates**. The reference is
`AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus_grid`.
Only its REPA feature loading, auxiliary projection/loss and diagnostics are
ported; this model keeps the source AVT joint attention and fixed temporal band.
Copied evaluation reports describe the parent experiment without REPA.

## Preserved configuration

- 12 AVT joint-attention blocks + 6 audio-only blocks; dim 768, 12 heads.
- Fixed soft AV/VA Gaussian prior: sigma 0.150 s, offset 0, aligned 40-Hz grid.
  The nominal +/-6-frame core is not a hard attention cutoff.
- Full LibriTTS text-audio 50k EMA initialization, including trained text and
  CAM++ speaker parameters. Source checkpoint:
  `${ROOT_PREFIX}/zjw524/projects/alignDiT_idea6/libriSpeech_LibriTTS_pretrained_models/model_00050000.pt`.
  SHA256 `4ee7dd9e1377728a042e75ebbbe70c2a12cd830c9b078afe13ebca289d03503c`.
- All 802 parent tensors are transferred, with the original five vocabulary
  row mappings to 159 characters. Exactly six new REPA projector tensors are
  initialized. Online and EMA start with identical weights; EMA retains the
  parent step 50000, while GRID update and optimizer state start from zero.
- Same audited GRID cache, 29,531 train / 3,274 held-out validation utterances,
  full MPG audio, 64-D Semantic-VAE latents, 1024-D AV-HuBERT video, 192-D CAM++.
  Same character vocabulary and LibriSpeech latent normalization; no val training.
- Four RTX 4090 GPUs, BF16, 3,600 frames/GPU, max 32 samples, gradient accumulation 1,
  data workers 16, seed 666, LR 5e-5, LR warmup 20k, gradient clipping 1.
- CTC is zero through 10k, then rises linearly to 0.03 at 30k.
- **The original 1,000-epoch LR decay horizon is preserved.** Only the stopping
  condition changes from 200k to 150k; learning rate is not compressed into 150k.
  The actual four-rank sampler has 985 batches, 246 per rank, dropping one tail.
- Numbered checkpoints every 50k and latest checkpoint every 5k, in the new path.

## Added REPA objective

| Setting | Value |
| --- | --- |
| Teacher | Frozen `microsoft/wavlm-base-plus` |
| Teacher revision | `4c66d4806a428f2e922ccfa1a962776e232d487b` |
| Teacher checkpoint SHA256 | `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51` |
| Teacher features | Layer 12, 768-D FP16, about 50 Hz, complete unmasked canonical audio |
| Student features | Audio output of MM block 10, zero-based index 9 |
| Projector | 768 -> 2048 -> 2048 -> 768, SiLU, six parameter tensors |
| Alignment | Interpolate each utterance's valid teacher span to valid 40-Hz latent length |
| Loss | Mean `1-cosine` over generated valid frames only; no teacher gradients |
| Weight | 0.1 from the first update, with no new warmup |

Total loss is `flow_loss + scheduled_ctc_weight * ctc_loss + 0.1 * repa_loss`.
Reuse the reference's completed cache at
`${ROOT_PREFIX}/zjw524/projects/data/GRID_svae1000k_sample_seed666_fp32/repa`.
No online teacher model or additional teacher GPU memory is needed. Teacher
identity, extraction provenance, shapes and valid lengths are checked. Inference
loads projector parameters strictly but never calls the auxiliary projector;
sampling parameters and waveform decoding are unchanged.

## Verification and training

From this snapshot:

```bash
source env.sh
PYTHON="${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python"
PYTHONPATH=src "$PYTHON" -u scripts/test_grid_repa_transfer.py --real-parent --audit-cache
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=src "$PYTHON" -u scripts/check_grid_repa_gpu.py
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash src/aligndit/run/train/finetune_grid.sh \
  > logs/train_grid_window13_libritts_repa.log 2>&1 < /dev/null &
train_pid=$!
ps -o pid,ppid,sid,tty,stat,cmd -p "$train_pid"
```

The launcher checks the real configuration and cache, starts an isolated
TensorBoard service, and launches four BF16 workers. Default DDP port 29646;
TensorBoard selects a free port starting at 6030. Runtime overrides are
`NUM_PROCESSES`, `CUDA_VISIBLE_DEVICES`, `TRAIN_PORT`, `TENSORBOARD_PORT` and
`PYTHON_BIN`. Restart only this experiment with the same contract to resume.
The normal stop at 150k saves the numbered checkpoint before exiting.

The checkpoint directory is:

```text
${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_40hz_char
```

TensorBoard run directory, relative to this snapshot:

```text
runs/AlignDiT_MMDiT_c2_svae_speaker_avt_joint_fixed_band_window13_av_va_libritts_pretrained_repa_wavlm_base_plus_GRID_ctc003_warmup10k30k_semantic_vae_40hz_GRID_char
```

Scalars include total/flow/CTC/REPA losses, weighted losses, LR, global and
projector gradients. Exact process/port/logdir are recorded in
`logs/tensorboard_grid.json` and `logs/grid_training_run.json`; training contracts
and `parent_migration.json` are saved with the new checkpoints. The user deferred
IDE forwarding-link verification; local HTTP and scalar recording are checked.

## Evaluation after training

The copied GRID S1/S2 entries keep the same full validation protocol. Their
checkpoint defaults and output names are isolated from the no-REPA baseline.
Once 150k has completed, `scripts/evaluate_grid_150k.py` evaluates S1 and both S2
text variants in order, using all four GPUs. It is not automatically started by
the training command. Compare against the original 150k results using the same
protocol, seed, EMA, 32 NFE and CFG 5/2. This experiment tests REPA's effect; an
improvement in WER has not yet been established.
