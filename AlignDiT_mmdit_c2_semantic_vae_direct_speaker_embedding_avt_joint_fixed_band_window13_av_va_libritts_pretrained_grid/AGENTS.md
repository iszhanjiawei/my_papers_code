# GRID fixed-13 LibriTTS experiment

This independent snapshot targets GRID, as explicitly requested by the user.
The parent AGENTS.md rules for source isolation, Git, background jobs and
TensorBoard still apply. The original project must remain unchanged.

- Use `GRID_TRAINING.md`, `finetune_grid.yaml` and
  `src/aligndit/run/train/finetune_grid.sh` for this experiment. Inherited
  CelebV-Dub/LRS3 entries and evaluation reports describe other experiments.
- Preserve the copied AVT joint backbone, 12 multimodal + 6 audio-only blocks,
  fixed 13-frame soft AV/VA Gaussian prior, and full LibriTTS 50k EMA loading.
  GRID's adaptive-band/REPA project supplies only the dataset protocol.
- Reuse the complete audited GRID cache with its original train/val split,
  full MPG audio, 1024-D AV-HuBERT video, 64-D Semantic-VAE posterior samples
  and 192-D CAM++ embeddings. The validation split never enters training.
- Keep the pretrained character vocabulary and LibriSpeech normalization.
  Check GRID's actual extraction metadata, not the copied source VAE spec.
- Run with this snapshot's `PYTHONPATH=src`; keep outputs/checkpoints/events
  isolated from the source projects. Runtime artifacts are not Git content.
