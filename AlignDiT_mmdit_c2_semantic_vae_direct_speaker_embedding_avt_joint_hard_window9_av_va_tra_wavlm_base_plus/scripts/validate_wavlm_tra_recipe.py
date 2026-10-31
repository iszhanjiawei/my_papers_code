"""Check actual production configuration and inherited initial tensors on CPU."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.modules import PrecomputedAudioRepresentation
from f5_tts.model.utils import get_tokenizer


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT.parent / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_avt_joint_hard_window9_av_va"
SOURCE_CONFIG = "finetune_celebvdub_mm_c2_svae_speaker_avt_joint_hard_window9_av_va"
TARGET_CONFIG = SOURCE_CONFIG + "_tra_wavlm_base_plus"


def configuration(root, name):
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "src/aligndit/config")):
        return compose(config_name=name)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else key
        result.update(flatten(item, name) if isinstance(item, dict) else {name: item})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "logs/tra_recipe_validation.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_config, target_config = configuration(SOURCE, SOURCE_CONFIG), configuration(PROJECT, TARGET_CONFIG)
    source_values = flatten(OmegaConf.to_container(source_config, resolve=True))
    target_values = flatten(OmegaConf.to_container(target_config, resolve=True))
    differing = {
        key: {"baseline": source_values.get(key), "tra": target_values.get(key)}
        for key in source_values.keys() | target_values.keys()
        if source_values.get(key) != target_values.get(key)
    }
    allowed = {
        "model.name",
        "ckpts.save_dir",
        "model.repa_lambda",
        "model.repa_schedule",
        "model.repa_schedule_eps",
        "model.arch.repa_layer",
        "model.arch.repa_target_dim",
        "model.arch.repa_projector_dim",
        "datasets.repa_feature_cache_dir",
        "datasets.repa_feature_dim",
        "datasets.repa_model_id",
        "datasets.repa_model_revision",
        "datasets.repa_checkpoint_sha256",
        "datasets.repa_teacher_layer",
    }
    assert set(differing) == allowed, sorted(set(differing) ^ allowed)
    vocab, size = get_tokenizer(target_config.datasets.vocab_path, "custom")
    assert size == 159, size
    target_cls = hydra.utils.get_class(f"aligndit.model.{target_config.model.backbone}")
    spec = importlib.util.spec_from_file_location(
        "original_hard9_backbone_recipe", SOURCE / "src/aligndit/model/backbone/dit_vt_mm.py"
    )
    source_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source_module)
    source_cls = source_module.DiT_VT_MMDiT

    def make_model(config, cls):
        torch.manual_seed(int(config.seed))
        audio = config.model.audio_representation
        return CFM_VT(
            transformer=cls(**config.model.arch, text_num_embeds=size, mel_dim=audio.channels),
            mel_spec_module=PrecomputedAudioRepresentation(audio.channels, audio.sample_rate, audio.hop_length),
            num_channels=audio.channels,
            vocab_char_map=vocab,
            audio_video_ratio=config.model.arch.audio_video_ratio,
            ctc_lambda=config.model.ctc_lambda,
            repa_lambda=config.model.get("repa_lambda", 0.0),
            repa_schedule=config.model.get("repa_schedule", "none"),
            repa_schedule_eps=config.model.get("repa_schedule_eps", 1e-8),
        )

    baseline = make_model(source_config, source_cls)
    source_rng = torch.random.get_rng_state().clone()
    source_count = sum(p.numel() for p in baseline.parameters())
    baseline_state = baseline.state_dict()
    assert source_count == 376563842, source_count
    assert len(baseline_state) == 802, len(baseline_state)
    # Target implementation with the original recipe adds no tensors or RNG draws.
    disabled = make_model(source_config, target_cls)
    assert torch.equal(source_rng, torch.random.get_rng_state())
    assert baseline_state.keys() == disabled.state_dict().keys()
    for key, value in disabled.state_dict().items():
        assert torch.equal(value, baseline_state[key]), key
    del disabled
    gc.collect()
    target = make_model(target_config, target_cls)
    target_state = target.state_dict()
    for key, value in baseline_state.items():
        assert torch.equal(value, target_state[key]), key
    added = sorted(set(target_state) - set(baseline_state))
    assert len(added) == 6 and all(k.startswith("transformer.repa_projector.") for k in added)
    target_count = sum(p.numel() for p in target.parameters())
    projector_count = sum(p.numel() for p in target.transformer.repa_projector.parameters())
    assert projector_count == 7344896, projector_count
    assert target_count == 383908738, target_count
    assert len(target_state) == 808, len(target_state)
    assert target.transformer.n_mm_layers == 12
    assert len(target.transformer.transformer_blocks) == 18
    assert target.transformer.temporal_band.window_radius_frames == 4
    result = {
        "complete": True,
        "baseline_parameters": source_count,
        "target_parameters": target_count,
        "projector_parameters": projector_count,
        "baseline_state_keys": len(baseline_state),
        "target_state_keys": len(target_state),
        "inherited_tensors_bitwise_equal": True,
        "disabled_rng_and_tensors_bitwise_equal": True,
        "vocab_size": size,
        "architecture": {
            "dim": target_config.model.arch.dim,
            "mm_layers": 12,
            "audio_tail_layers": 6,
            "hard_window_radius_frames": 4,
            "repa_layer": 9,
        },
        "added_keys": added,
        "only_intended_configuration_changes": differing,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
