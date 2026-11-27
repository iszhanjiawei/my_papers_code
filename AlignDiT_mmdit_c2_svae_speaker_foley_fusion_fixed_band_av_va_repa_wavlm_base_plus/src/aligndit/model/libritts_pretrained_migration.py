"""Load Foley audio/text EMA into CelebV-Dub with character-aware output rows."""

import json
from pathlib import Path

import torch

from aligndit.model.semantic_vae_direct_migration import sha256_file
from aligndit.model.text_audio_checkpoint import contract_sha256


def read_vocabulary(path, expected_digest):
    if sha256_file(path) != expected_digest:
        raise ValueError(f"Vocabulary hash differs from training contract: {path}")
    tokens = Path(path).read_text(encoding="utf-8").splitlines()
    if len(tokens) != len(set(tokens)) or any(len(token) != 1 for token in tokens):
        raise ValueError(f"Expected unique characters in {path}")
    return tokens


def load_libritts_ema_into_model(model, checkpoint_path, config):
    path = Path(checkpoint_path)
    ckpts, data = config["ckpts"], config["datasets"]
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    step = ckpts["expected_parent_update"]
    if (payload.get("checkpoint_schema_version") != 1 or payload.get("stage") != "libritts_text_audio"
            or payload.get("update") != step or step <= 0):
        raise ValueError("Unexpected LibriTTS checkpoint stage or update")
    source = payload["training_contract"]
    digest = contract_sha256(source)
    if (digest != payload["training_contract_sha256"] or source.get("stage") != "libritts_text_audio"
            or source["model"]["name"] != ckpts["libritts_run_name"]
            or source["initialization"]["type"] != "s2c_ema"
            or source["initialization"]["parent_sha256"] != ckpts["s2c_sha256"]):
        raise ValueError("LibriTTS run identity or S2c initialization differs")
    arch = source["model"]["arch"]
    if not (arch.get("text_audio_only") and arch.get("joint_av_text_ca")) or arch.get("temporal_band_enabled"):
        raise ValueError("Expected text/audio pretraining of the Foley cross-attention")
    target_model = config["model"]
    for key in ("depth", "dim", "heads", "text_dim", "ff_mult", "n_mm_layers", "n_text_layers",
                "ctc_sampling_ratios", "layer_indices_ctc", "qk_norm", "pe_attn_head", "conv_layers",
                "prompt_isolated_ca", "speaker_dim", "speaker_condition_start_layer", "joint_av_text_ca"):
        if arch[key] != target_model["arch"][key]:
            raise ValueError(f"Incompatible text/audio architecture: {key}")
    if source["model"]["audio_representation"] != target_model["audio_representation"]:
        raise ValueError("Incompatible Semantic-VAE audio representation")
    spec = json.loads((Path(data["cache_root"]) / "state/latents/spec.json").read_text())
    required = {"normalization_sha256": data["expected_normalization_sha256"],
                "speaker_checkpoint_sha256": data["speaker_embedding_checkpoint_sha256"],
                "posterior_protocol": spec["extraction"]["protocol"],
                "codec_sha256": spec["checkpoint"]["ema_sha256"]}
    for key, expected in required.items():
        if source["data"].get(key) != expected:
            raise ValueError(f"Incompatible text/audio feature contract: {key}")
    source_tokens = read_vocabulary(ckpts["libritts_vocab_path"], source["data"]["vocab_sha256"])
    target_tokens = read_vocabulary(data["vocab_path"], data["expected_vocab_sha256"])
    source_index = {ch: index for index, ch in enumerate(source_tokens)}
    if not set(target_tokens).issubset(source_index):
        raise ValueError("LibriTTS vocabulary does not cover the downstream characters")
    token_rows = [source_index[ch] for ch in target_tokens]
    embedding_rows = [0] + [i + 1 for i in token_rows]  # row zero is filler
    ctc_rows = token_rows + [len(source_tokens), len(source_tokens) + 1]  # unused and blank
    remaps = {"transformer.text_embed.text_embed.weight": embedding_rows}
    for index, projector in enumerate(model.transformer.projectors_ctc):
        output_index = len(projector.model) - 1
        for suffix in ("weight", "bias"):
            remaps[f"transformer.projectors_ctc.{index}.model.{output_index}.{suffix}"] = ctc_rows

    ema = payload["ema_model_state_dict"]
    if not bool(ema.get("initted")) or int(ema["step"]) != step:
        raise ValueError("LibriTTS EMA is not initialized at the requested update")
    weights = {k.removeprefix("ema_model."): v for k, v in ema.items() if k not in {"step", "initted"}}
    target = model.state_dict()
    new_keys = {key for key in target if key.startswith("transformer.repa_projector.")}
    if set(weights) != set(target) - new_keys or not remaps.keys() <= weights.keys():
        raise ValueError("Source and destination should differ only by the downstream REPA head")
    for key, rows in remaps.items():
        source_rows = len(source_tokens) + (1 if key.endswith("text_embed.weight") else 2)
        if weights[key].shape[0] != source_rows:
            raise ValueError(f"Wrong number of LibriTTS vocabulary rows in {key}")
        weights[key] = weights[key].index_select(0, torch.tensor(rows, dtype=torch.long))
    for key, value in weights.items():
        if value.shape != target[key].shape:
            raise ValueError(f"Incompatible LibriTTS EMA tensor shape for {key}")
    model.load_state_dict({**target, **weights}, strict=True)
    return {"parent_path": str(path.resolve()), "parent_stage": source["stage"], "parent_update": step,
            "parent_ema_step": step, "parent_contract_sha256": digest, "loaded_key_count": len(weights),
            "new_target_keys": sorted(new_keys), "row_remappings": remaps,
            "visual_weights_pretrained": False, "source_ctc_lambda": source["model"]["ctc_lambda"],
            "optimizer_restored": False}
