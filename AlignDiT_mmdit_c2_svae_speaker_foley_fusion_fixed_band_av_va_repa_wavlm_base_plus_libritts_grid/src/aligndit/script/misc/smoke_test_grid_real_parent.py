"""Exercise the GRID training model on two real utterances and the LibriTTS EMA."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
from accelerate.utils import set_seed
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.libritts_pretrained_migration import load_libritts_ema_into_model
from aligndit.script.train.finetune_grid import dataset_from_config, model_from_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", default="finetune_grid")
    args = parser.parse_args()
    with initialize_config_dir(version_base="1.3", config_dir=str(Path(__file__).resolve().parents[2] / "config")):
        cfg = compose(config_name=args.config_name)
    set_seed(cfg.seed)
    dataset = dataset_from_config(cfg)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    model = model_from_config(cfg)
    migration = load_libritts_ema_into_model(model, cfg.ckpts.pretrained_path, OmegaConf.to_container(cfg, resolve=True))
    print(f"Loaded {migration['loaded_key_count']} EMA tensors from step {migration['parent_update']}", flush=True)
    device = torch.device("cuda")
    model.to(device).train()
    for ctc_weight in (0.0, cfg.model.ctc_lambda):
        model.ctc_lambda = ctc_weight
        model.zero_grad(set_to_none=True)
        # Exercise text, video and speaker together even when CFG dropout is configured.
        with patch("aligndit.model.cfm_vt.random", return_value=0.99), torch.autocast("cuda", dtype=torch.bfloat16):
            loss, parts, _, prediction = model(
                inp=batch["mel"].transpose(1, 2).to(device),
                text=batch["text"],
                lens=batch["mel_lengths"].to(device),
                text_lens=batch["text_lengths"].to(device),
                video=batch["video"].to(device),
                video_lens=batch["video_lengths"].to(device),
                speaker_embedding=batch["speaker_embedding"].to(device),
                repa_features=batch["repa_features"].to(device),
                repa_feature_lens=batch["repa_feature_lengths"].to(device),
            )
        assert torch.isfinite(loss) and torch.isfinite(prediction).all()
        assert 0 <= parts["repa_loss"] <= 2
        if ctc_weight:
            assert parts["ctc_loss"] > 0
        loss.backward()
        block = model.transformer.transformer_blocks[0]
        # Video CA starts with a zero gate. The first backward pass trains the
        # gate itself before its output projection starts receiving gradients.
        projections = {
            "audio Query": block.cross_attn.audio_cross_q,
            "text Key/Value": block.cross_attn.text_cross_kv,
            "video CA gate": block.v_cross_attn_ada,
            "speaker": model.transformer.speaker_proj,
            "REPA": model.transformer.repa_projector[-1],
        }
        for name, projection in projections.items():
            grad = projection.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.norm() > 0, name
        print(f"{batch['utterance_keys']} CTC weight={ctc_weight} loss={loss.item():.6f} {parts}", flush=True)


if __name__ == "__main__":
    main()
