"""Migrate the pinned LibriTTS text/audio EMA into the concat-text Foley backbone.

Keep shared audio, text encoder, video, speaker and CTC tensors. Only the
obsolete joint-text stream is discarded. New modules keep their own initial
values and EMA starts from the resulting online state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from aligndit.model.semantic_vae_direct_migration import sha256_file


PARENT_SHA256 = "4ee7dd9e1377728a042e75ebbbe70c2a12cd830c9b078afe13ebca289d03503c"
PARENT_SIZE = 5048750348
PARENT_UPDATE = 50000
PARENT_CONTRACT_SHA256 = "88e16bfe6a0c393bd4236e70a1bebd824713cc00c06bbb709e3a530d433f236f"
TARGET_VOCAB_SHA256 = "225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d"
SOURCE_VOCAB_SHA256 = "7032586c60c7d17d17e0d580f8cdb05b5652326e43a287f405367dcacfc74ac0"
# Reconstructed from all 149,736 normalized LibriTTS-460 transcripts and checked
# against the vocabulary hash embedded in the checkpoint's training contract.
SOURCE_EXTRA_TOKENS = ("(", ")", "/", ":", ";", "[", "]", "{", "}", "ê", "œ", "\u0304")
NORMALIZATION_SHA256 = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
CODEC_SHA256 = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
SPEAKER_SHA256 = "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199"
EMBEDDING_KEY = "transformer.text_embed.text_embed.weight"
CTC_ROW_KEYS = frozenset(
    f"transformer.projectors_ctc.{index}.model.6.{suffix}"
    for index in (0, 1) for suffix in ("weight", "bias")
)
REPA_PROJECTOR_SHAPES = {
    "transformer.repa_projector.0.weight": (2048, 768),
    "transformer.repa_projector.0.bias": (2048,),
    "transformer.repa_projector.2.weight": (2048, 2048),
    "transformer.repa_projector.2.bias": (2048,),
    "transformer.repa_projector.4.weight": (768, 2048),
    "transformer.repa_projector.4.bias": (768,),
}


def _plain(config):
    from omegaconf import OmegaConf

    return OmegaConf.to_container(config, resolve=True) if OmegaConf.is_config(config) else config


def _identity(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _validate_pins(config):
    expected = {
        "expected_parent_sha256": PARENT_SHA256,
        "expected_parent_size": PARENT_SIZE,
        "expected_parent_update": PARENT_UPDATE,
        "expected_parent_contract_sha256": PARENT_CONTRACT_SHA256,
    }
    for key, value in expected.items():
        if config["ckpts"].get(key) != value:
            raise ValueError(f"LibriTTS parent identity is not pinned correctly: {key}")


def validate_libritts_checkpoint_artifact(checkpoint_path, model_cfg) -> dict:
    """Hash the immutable parent once on rank zero before distributed loading."""
    config = _plain(model_cfg)
    _validate_pins(config)
    path = Path(checkpoint_path).expanduser().absolute()
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"LibriTTS parent must be a regular file: {path}")
    identity = _identity(path)
    if identity[2] != PARENT_SIZE:
        raise ValueError("LibriTTS parent checkpoint size mismatch")
    actual = sha256_file(path)
    if actual != PARENT_SHA256:
        raise ValueError("LibriTTS parent checkpoint SHA256 mismatch")
    if _identity(path) != identity:
        raise RuntimeError("LibriTTS parent changed while its hash was being checked")
    return {"path": str(path.resolve()), "sha256": actual, "size": identity[2]}


def build_vocabulary_mapping(vocab_path, expected_target_sha256=TARGET_VOCAB_SHA256) -> dict:
    """Verify both vocabularies and return embedding/CTC semantic row maps."""
    payload = Path(vocab_path).read_bytes()
    if expected_target_sha256 != TARGET_VOCAB_SHA256 or hashlib.sha256(payload).hexdigest() != TARGET_VOCAB_SHA256:
        raise ValueError("Downstream CelebVDub vocabulary SHA256 mismatch")
    target = payload.decode("utf-8").splitlines()
    if len(target) != 159 or len(set(target)) != 159 or any(len(token) != 1 for token in target):
        raise ValueError("Expected the unchanged 159-character CelebVDub vocabulary")
    source = target + list(SOURCE_EXTRA_TOKENS)
    source_payload = ("\n".join(source) + "\n").encode("utf-8")
    if hashlib.sha256(source_payload).hexdigest() != SOURCE_VOCAB_SHA256:
        raise ValueError("Reconstructed LibriTTS vocabulary does not match its embedded SHA256")
    source_index = {token: index for index, token in enumerate(source)}
    # TextEmbedding shifts characters by one and reserves zero for filler.
    embedding_rows = [0] + [source_index[token] + 1 for token in target]
    # CTC consumes unshifted character IDs. Its unused output row and final blank
    # follow the vocabulary, so these two rows must move with the output size.
    ctc_rows = [source_index[token] for token in target] + [len(source), len(source) + 1]
    return {
        "target_vocab_sha256": TARGET_VOCAB_SHA256,
        "source_vocab_sha256": SOURCE_VOCAB_SHA256,
        "target_tokens": target,
        "source_tokens": source,
        "source_extra_tokens": list(SOURCE_EXTRA_TOKENS),
        "embedding_source_rows": embedding_rows,
        "ctc_source_rows": ctc_rows,
        "embedding_filler": {"target": 0, "source": 0},
        "ctc_unused": {"target": len(target), "source": len(source)},
        "ctc_blank": {"target": len(target) + 1, "source": len(source) + 1},
        "source_vocab_reconstruction": "CelebVDub base plus sorted LibriTTS-460 transcript additions; full hash verified",
    }


def _validate_transfer_contract(source, config):
    if source.get("contract_schema_version") != 1 or source.get("stage") != "libritts_text_audio":
        raise ValueError("Unsupported LibriTTS embedded training contract")
    src, dst = source["model"], config["model"]
    if src.get("backbone") != "DiT_VT_MMDiT" or dst.get("backbone") != "DiT_VT_MMDiT":
        raise ValueError("LibriTTS transfer backbone mismatch")
    required = {
        "dim": 768, "depth": 18, "heads": 12, "ff_mult": 2, "text_dim": 512,
        "qk_norm": "rms_norm", "conv_layers": 4, "pe_attn_head": 1,
        "attn_mask_enabled": True, "use_conformer": True,
        "layer_indices_ctc": [6, 12], "ctc_sampling_ratios": [1, 1],
        "n_mm_layers": 12, "n_text_layers": 12, "prompt_isolated_ca": False,
        "audio_video_ratio": 1, "video_dim": 1024, "video_rope_scaled": False,
        "speaker_dim": 192, "speaker_condition_start_layer": 12,
    }
    for key, value in required.items():
        if src["arch"].get(key) != value or dst["arch"].get(key) != value:
            raise ValueError(f"LibriTTS transfer architecture mismatch: {key}")
    for key, default in (("dim_head", 64), ("text_embedding_average_upsampling", False),
                         ("normalize_text_context", False)):
        if src["arch"].get(key, default) != default or dst["arch"].get(key, default) != default:
            raise ValueError(f"LibriTTS transfer architecture mismatch: {key}")
    if src["arch"].get("text_joint_attention") is not True or src["arch"].get("text_audio_only") is not True:
        raise ValueError("Expected the original LibriTTS joint text/audio stage")
    if not src["arch"].get("text_mask_padding") or src["arch"].get("temporal_band_enabled") is not False:
        raise ValueError("Unexpected LibriTTS text-mask or temporal prior")
    required_target = {
        "joint_av_text_ca": True, "role_conditioned_text": False, "text_mask_padding": False, "temporal_band_enabled": True,
        "temporal_band_mode": "fixed", "temporal_band_fixed_sigma_seconds": 0.1,
        "temporal_band_fixed_offset_seconds": 0.0, "temporal_band_audio_fps": 40.0,
        "temporal_band_video_fps": 40.0, "temporal_band_bidirectional": True,
    }
    for key, value in required_target.items():
        if dst["arch"].get(key) != value:
            raise ValueError(f"Target concat-text/fusion configuration mismatch: {key}")
    for key, value in {"channels": 64, "frame_rate": 40, "sample_rate": 16000, "hop_length": 400}.items():
        if src["audio_representation"].get(key) != value or dst["audio_representation"].get(key) != value:
            raise ValueError(f"LibriTTS transfer audio representation mismatch: {key}")
    data = config["datasets"]
    required_data = {"normalization_sha256": NORMALIZATION_SHA256, "codec_sha256": CODEC_SHA256,
                     "semantic_vae_checkpoint_sha256": CODEC_SHA256,
                     "speaker_checkpoint_sha256": SPEAKER_SHA256, "vocab_sha256": SOURCE_VOCAB_SHA256}
    for key, value in required_data.items():
        if source["data"].get(key) != value:
            raise ValueError(f"LibriTTS source representation contract mismatch: {key}")
    if data.get("expected_normalization_sha256") != NORMALIZATION_SHA256:
        raise ValueError("Downstream normalization configuration changed")
    if sha256_file(data["normalization_path"]) != NORMALIZATION_SHA256:
        raise ValueError("Downstream normalization file changed")
    if data.get("speaker_embedding_checkpoint_sha256") != SPEAKER_SHA256:
        raise ValueError("Downstream speaker encoder contract changed")
    spec = json.loads((Path(data["cache_root"]) / "state/latents/spec.json").read_text())
    if spec["checkpoint"].get("ema_sha256") != CODEC_SHA256:
        raise ValueError("Downstream latent cache codec differs from the LibriTTS parent")
    if spec["extraction"].get("protocol") != source["data"].get("posterior_protocol"):
        raise ValueError("Source and target latent posterior protocols differ")


def _validate_model(model, config):
    backbone, arch = model.transformer, config["model"]["arch"]
    for key in ("dim", "depth", "n_mm_layers", "n_text_layers", "joint_av_text_ca", "role_conditioned_text",
                "speaker_dim", "speaker_condition_start_layer", "audio_video_ratio", "video_rope_scaled"):
        if getattr(backbone, key, None) != arch[key]:
            raise ValueError(f"Instantiated backbone differs from target configuration: {key}")
    if backbone.text_embed.text_embed.num_embeddings != 160:
        raise ValueError("Instantiated model must retain the downstream 160-row text embedding")
    for key, value in {"repa_layer": 9, "repa_target_dim": 768}.items():
        if arch.get(key) != value or getattr(backbone, key, None) != value:
            raise ValueError(f"Instantiated REPA model/configuration mismatch: {key}")
    if arch.get("repa_projector_dim") != 2048 or float(config["model"].get("repa_lambda", 0.0)) != 0.1:
        raise ValueError("LibriTTS transfer requires the pinned REPA projector and loss weight")
    if getattr(model, "repa_lambda", None) != 0.1:
        raise ValueError("Instantiated CFM must enable REPA with lambda=0.1")


def _expected_unshared_keys():
    # The parent updates text as a third stream through twelve joint blocks.
    # None of those projections has an equivalent in the new static text KV path.
    ignored = {"transformer.text_joint_proj.weight", "transformer.text_joint_proj.bias"}
    fresh = set(REPA_PROJECTOR_SHAPES)
    for layer in range(12):
        block = f"transformer.transformer_blocks.{layer}."
        for module in ("t_attn_norm.linear", "t_attn.to_q", "t_attn.to_k", "t_attn.to_v", "t_attn.to_out.0",
                       "t_ff.ff.0.0", "t_ff.ff.2"):
            ignored.update(block + module + "." + suffix for suffix in ("weight", "bias"))
        ignored.update(block + "t_attn." + norm + ".weight" for norm in ("q_norm", "k_norm"))
        for module in ("cross_attn_ada", "v_cross_attn_ada", "cross_attn.audio_cross_q", "cross_attn.v_cond_cross_q",
                       "cross_attn.text_cross_kv", "cross_attn.audio_cross_proj", "cross_attn.v_cond_cross_proj"):
            fresh.update(block + module + "." + suffix for suffix in ("weight", "bias"))
        for norm in ("audio_cross_q_norm", "v_cond_cross_q_norm", "text_cross_k_norm"):
            fresh.add(block + "cross_attn." + norm + ".weight")
    return ignored, fresh


def validate_transfer_keys(source, target):
    ignored, fresh = _expected_unshared_keys()
    source_keys, target_keys = set(source), set(target)
    if len(source_keys) != 802:
        raise ValueError("Expected the 802 original LibriTTS EMA tensors")
    for label, actual, expected in (
        ("parent-only", source_keys - target_keys, ignored), ("target-only", target_keys - source_keys, fresh),
    ):
        if actual != expected:
            raise ValueError(f"Unexpected {label} tensors: missing={sorted(expected - actual)}, "
                             f"extra={sorted(actual - expected)}")
    return sorted(ignored), sorted(fresh)


def load_libritts_ema_into_model(model, checkpoint_path, model_cfg, *, verify_checkpoint_hash=True) -> dict:
    """Load matching EMA parameters and validate the exact architecture differences."""
    config = _plain(model_cfg)
    _validate_pins(config)
    path = Path(checkpoint_path).expanduser().absolute()
    if path.is_symlink() or not path.is_file() or path.stat().st_size != PARENT_SIZE:
        raise ValueError("LibriTTS parent file identity/size mismatch")
    before = _identity(path)
    if verify_checkpoint_hash:
        validate_libritts_checkpoint_artifact(path, config)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if checkpoint.get("checkpoint_schema_version") != 1 or checkpoint.get("stage") != "libritts_text_audio":
        raise ValueError("Unsupported LibriTTS checkpoint schema/stage")
    if checkpoint.get("update") != PARENT_UPDATE:
        raise ValueError("LibriTTS parent update must be exactly 50000")
    contract = checkpoint["training_contract"]
    contract_hash = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":"),
                                             ensure_ascii=False, allow_nan=False).encode()).hexdigest()
    if contract_hash != PARENT_CONTRACT_SHA256 or checkpoint.get("training_contract_sha256") != contract_hash:
        raise ValueError("LibriTTS embedded training contract SHA256 mismatch")
    _validate_transfer_contract(contract, config)
    _validate_model(model, config)
    vocabulary = build_vocabulary_mapping(config["datasets"]["vocab_path"], config["datasets"]["expected_vocab_sha256"])
    ema = checkpoint.get("ema_model_state_dict")
    if not isinstance(ema, dict) or set(ema) - {"initted", "step"} != {
        key for key in ema if key.startswith("ema_model.")
    }:
        raise ValueError("Unexpected LibriTTS EMA bookkeeping keys")
    if "initted" not in ema or "step" not in ema or not bool(ema["initted"]) or int(ema["step"]) != PARENT_UPDATE:
        raise ValueError("LibriTTS EMA must be initialized at step 50000")
    source = {key.removeprefix("ema_model."): value for key, value in ema.items() if key.startswith("ema_model.")}
    target = model.state_dict()
    ignored_keys, new_keys = validate_transfer_keys(source, target)
    remapped, loaded = {}, {}
    for key, destination in target.items():
        if key in new_keys:
            continue
        value = source[key]
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"LibriTTS source value is not a floating-point tensor: {key}")
        if key == EMBEDDING_KEY or key in CTC_ROW_KEYS:
            rows = vocabulary["embedding_source_rows" if key == EMBEDDING_KEY else "ctc_source_rows"]
            expected_source_rows = 172 if key == EMBEDDING_KEY else 173
            if value.shape[0] != expected_source_rows or value.shape[1:] != destination.shape[1:]:
                raise ValueError(f"Unexpected source vocabulary tensor shape: {key}")
            value = value.index_select(0, torch.tensor(rows, dtype=torch.long))
            remapped[key] = {"source_shape": list(source[key].shape), "target_shape": list(destination.shape),
                             "source_rows": rows}
        if value.shape != destination.shape:
            raise ValueError(f"LibriTTS transfer tensor shape mismatch: {key}")
        loaded[key] = value
    if set(remapped) != {EMBEDDING_KEY, *CTC_ROW_KEYS}:
        raise RuntimeError("LibriTTS transfer did not perform exactly the five approved row remappings")
    if _identity(path) != before:
        raise RuntimeError("LibriTTS parent changed while its tensors were being read")
    model.load_state_dict({**loaded, **{key: target[key] for key in new_keys}}, strict=True)
    return {
        "initialization": "LibriTTS 50k EMA shared audio, video, text encoder, speaker and CTC parameters",
        "parent_path": str(path.resolve()), "parent_sha256": PARENT_SHA256, "parent_size": PARENT_SIZE,
        "parent_contract_sha256": contract_hash, "parent_stage": "libritts_text_audio",
        "parent_update": PARENT_UPDATE, "parent_ema_step": PARENT_UPDATE,
        "source_key_count": len(source), "target_key_count": len(target), "loaded_key_count": len(loaded),
        "shape_identical_key_count": len(loaded) - len(remapped), "ignored_source_keys": ignored_keys, "new_target_keys": new_keys,
        "new_target_key_count": len(new_keys),
        "new_target_shapes": {key: list(target[key].shape) for key in new_keys},
        "new_target_numel": sum(target[key].numel() for key in new_keys),
        "new_target_initialization": "fresh Foley joint-query text attention, modulation and REPA; identical online/EMA initialization",
        "loaded_keys": sorted(loaded), "loaded_numel": sum(value.numel() for value in loaded.values()),
        "row_remappings": remapped, "vocabulary": vocabulary,
        "source_text_audio_only": True, "target_text_audio_only": False,
        "intentional_runtime_changes": ["one text stream for Foley A/V queries", "concatenated reference and target transcript at inference",
                                        "bidirectional fixed 100 ms Gaussian", "masked-span WavLM-Base+ REPA"],
        "source_ctc_lambda": contract["model"]["ctc_lambda"],
        "visual_weights_pretrained": False, "ctc_weights_pretrained": False,
        "speaker_weights_pretrained": True, "optimizer_restored": False, "downstream_initial_update": 0,
        "ema_policy": "Carry parent EMA step 50000, with fresh downstream update counters",
        "function_preserving": False,
    }
