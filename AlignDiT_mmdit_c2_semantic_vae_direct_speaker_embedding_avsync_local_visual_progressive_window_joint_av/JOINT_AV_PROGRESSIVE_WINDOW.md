# Joint Audio-query / Video-key progressive-window experiment

This directory is an independent source copy of `AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avsync_local_visual_progressive_window` at repository commit `ee8e79d`. Its runtime logs, TensorBoard events and checkpoints are excluded. All modifications and new outputs belong to this copy.

The user-selected protocol retains the 12 multimodal + 6 audio-only architecture and the six existing gated visual adapters. The addition is Flowley's time bias inside the Audio-query -> Video-key quadrant of the first twelve blocks' joint attention. Both groups retain the 0.5-second outer radius and use independent progressive fade amplitudes; the radius does not shrink with depth or training time.

| Attention location | Zero-based blocks | Fade amplitude beta |
|---|---|---|
| Joint Audio-query -> Video-key | 0..11 | `1 - layer/11` |
| Gated audio-tail visual attention | 12..17 | `1 - (layer-12)/5` |

## Attention semantics

Both locations reuse the parameter-free `FlowleyTemporalWindow` geometry. At 40 Hz, the 0.5-second radius spans 20 tokens on each side (4 reference frames at 8 Hz). The core radius remains 0. Outside the core, the cosine fade is multiplied by beta; the attention bias is `log(weight + 1e-6)`. The last block of each group has beta=0: the center retains weight 1, and other positions have the epsilon floor.

The concatenated joint key/value sequence remains `[audio, video]`, with one shared softmax normalization for each query. Only generated-audio queries paired with real visual keys receive the extra time bias. The real-video mask excludes prompt/null positions and padding, and is cleared for video-dropped CFG branches. Other query/key pairs receive zero extra bias. Existing joint padding behavior is retained, including the parent's omission of ordinary joint padding masks during training; a separate real-video mask supplies valid geometry in both training and inference. Full timeline coordinates are preserved: prompt-prefix holes are not removed, and valid endpoints use the last real video position rather than its count.

Audio-query -> Audio-key, Video-query -> Audio-key, and Video-query -> Video-key logits do not receive a temporal bias. Changing Audio-query -> Video-key logits still changes the relative audio/video probability mass through the shared softmax. At identical inputs, video-query attention is unchanged. Audio/video states can already contain context from preceding layers; the constraint applies to the direct attention positions.

The joint path retains its Q/K/V projections, Q/K normalization, RoPE and existing AdaLN output gates. It adds no learnable parameters or persistent checkpoint tensors. The final six visual residuals still have independent per-channel gates initialized to `1e-5`. Audio and video query rows are evaluated in separate equivalent SDPA calls in the enabled joint path, limiting the additive float mask to `[batch, 1, audio_length, audio_length + video_length]`.

`joint_av_local_attention` defaults to false, preserving parent behavior. Its `joint_av_window_schedule` accepts `fixed` or `flowley_progressive`, independently of `local_visual_window_schedule` for the tail. Joint geometry uses the existing `audio_frame_rate`, `audio_video_ratio`, and `local_visual_window_*` settings. The new YAML enables both progressive schedules. Use this YAML for training, resume and inference: compatible parameter shapes alone do not encode the attention window policy.

## Training

The new config is `src/aligndit/config/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_joint_av_progressive_window_ctc003_warmup.yaml`. It inherits the parent config and changes only the two joint-attention options, experiment name and checkpoint directory. CAM++ conditioning, CTC taps [6, 12], seed 666, S2c 70k pure-audio EMA initialization, LR=5e-5, bf16, 3600 frames/GPU and the target 200k updates are retained. Optimizer and update counter start fresh. CTC is disabled through 10k and warms linearly to 0.03 at 30k.

Run from this project on four free RTX 4090 GPUs:

```bash
mkdir -p logs
setsid env PYTHONUNBUFFERED=1 bash \
  src/aligndit/run/train/finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_joint_av_progressive_window_4x4090.sh \
  > logs/train_joint_av_progressive_window.log 2>&1 < /dev/null &
```

The default training port is 29621 (`TRAIN_PORT` can override it). Checkpoints use `${ROOT_PREFIX}/zjw524/projects/data/ckpts/AlignDiT_MMDiT_c2_svae_speaker_jointav_local_visual_flowley_progressive_ctc003_warmup10k30k_40hz_CelebVDub_char`. Numbered checkpoints are saved every 50k updates; a rolling checkpoint is saved every 5k.

The training contract records both sets of actual per-layer window values, direction, active positions and gate semantics. Main-rank TensorBoard events record `loss`, `diff_loss`, `ctc_lambda`, `ctc_weighted_loss`, gradient diagnostics and tail visual gates. Raw `ctc_loss` starts after 10k when CTC computation begins.

```bash
TENSORBOARD_LOGDIR="$PWD/runs/AlignDiT_MMDiT_c2_svae_speaker_jointav_local_visual_flowley_progressive_ctc003_warmup10k30k_semantic_vae_40hz_CelebVDub_char" \
  TENSORBOARD_PORT=6006 bash scripts/start_speaker_tensorboard.sh
```

Verify the chosen port is free. The launcher creates a detached TensorBoard service. The client can forward this actual port from its Ports panel and open the displayed forwarding link. Current PID, event path and checked endpoint belong in ignored `logs/` runtime records.

## Validation and inference

```bash
source env.sh
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" scripts/test_audio_local_visual_attention.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" scripts/test_progressive_visual_window.py
PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" scripts/test_joint_av_progressive_attention.py
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src "${ROOT_PREFIX}/zjw524/ENTER/envs/aligndit/bin/python" -u \
  src/aligndit/script/misc/smoke_test_semantic_vae_c2_speaker_real_parent.py \
  --config-name finetune_celebvdub_mm_c2_semantic_vae_direct_speaker_joint_av_progressive_window_ctc003_warmup
```

The GPU smoke validates the real S2c checkpoint identity and exact migrated tensors, both schedules, real cached samples, and bf16 forward/backward with CTC weights 0 and 0.03. Strict migration remains source=313, target=770, loaded=303, ignored=10 and new=467; the new joint bias requires no relaxed migration rules. The smoke does not perform optimizer updates.

For a completed checkpoint, run `CKPT_STEP=150000 EVAL_GPU=0 bash src/aligndit/run/eval/eval_celebvdub_s1_svae_speaker_joint_av_progressive_window.sh`. It selects the new YAML and checkpoint directory, then executes the inherited CelebV-Dub Setting 1 inference/four-metric protocol. Use `infer_celebvdub_s1_svae_speaker_joint_av_progressive_window.sh` for inference alone.
