"""Load the pinned LibriTTS EMA while preserving the fixed-13 CelebVDub recipe.

The LibriTTS vocabulary extends the original CelebVDub vocabulary. Transfer
embedding and CTC rows by their meanings; never change the downstream vocabulary.
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
        "text_joint_attention": True, "speaker_dim": 192, "speaker_condition_start_layer": 12,
    }
    for key, value in required.items():
        if src["arch"].get(key) != value or dst["arch"].get(key) != value:
            raise ValueError(f"LibriTTS transfer architecture mismatch: {key}")
    for key, default in (("dim_head", 64), ("text_embedding_average_upsampling", False),
                         ("normalize_text_context", False)):
        if src["arch"].get(key, default) != default or dst["arch"].get(key, default) != default:
            raise ValueError(f"LibriTTS transfer architecture mismatch: {key}")
    if src["arch"].get("text_audio_only") is not True or dst["arch"].get("text_audio_only", False) is not False:
        raise ValueError("Expected explicit LibriTTS AT-to-CelebVDub AVT initialization")
    if src["arch"].get("text_mask_padding") is not True or dst["arch"].get("text_mask_padding") is not False:
        raise ValueError("The original downstream text_mask_padding=False must be retained")
    if src["arch"].get("temporal_band_enabled") is not False:
        raise ValueError("Unexpected temporal prior in the LibriTTS parent")
    for key, value in {"temporal_band_enabled": True, "temporal_band_mode": "fixed",
                       "temporal_band_fixed_sigma_seconds": 0.150,
                       "temporal_band_fixed_offset_seconds": 0.0,
                       "temporal_band_audio_fps": 40.0, "temporal_band_video_fps": 40.0,
                       "temporal_band_bidirectional": True}.items():
        if dst["arch"].get(key) != value:
            raise ValueError(f"The original fixed-13 temporal prior must be retained: {key}")
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
    for key in ("dim", "depth", "n_mm_layers", "n_text_layers", "text_joint_attention",
                "speaker_dim", "speaker_condition_start_layer", "audio_video_ratio", "video_rope_scaled"):
        if getattr(backbone, key) != arch[key]:
            raise ValueError(f"Instantiated model differs from the fixed-13 configuration: {key}")
    if getattr(backbone, "text_audio_only", False) or backbone.text_embed.mask_padding:
        raise ValueError("Instantiated model must retain AVT and text_mask_padding=False")
    if backbone.text_embed.text_embed.num_embeddings != 160:
        raise ValueError("Instantiated model must retain the downstream 160-row text embedding")


def load_libritts_ema_into_model(model, checkpoint_path, model_cfg, *, verify_checkpoint_hash=True) -> dict:
    """Load all 802 EMA tensors, with five explicit vocabulary row remappings."""
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
    if len(source) != 802 or len(target) != 802 or set(source) != set(target):
        raise ValueError("LibriTTS transfer requires exactly the same 802 tensor names")
    remapped, loaded = {}, {}
    for key, destination in target.items():
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
    model.load_state_dict(loaded, strict=True)
    return {
        "initialization": "Complete LibriTTS text/audio/speaker EMA; original fixed-13 AVT configuration",
        "parent_path": str(path.resolve()), "parent_sha256": PARENT_SHA256, "parent_size": PARENT_SIZE,
        "parent_contract_sha256": contract_hash, "parent_stage": "libritts_text_audio",
        "parent_update": PARENT_UPDATE, "parent_ema_step": PARENT_UPDATE,
        "source_key_count": len(source), "target_key_count": len(target), "loaded_key_count": len(loaded),
        "shape_identical_key_count": len(loaded) - len(remapped), "ignored_source_keys": [], "new_target_keys": [],
        "loaded_keys": sorted(loaded), "loaded_numel": sum(value.numel() for value in loaded.values()),
        "row_remappings": remapped, "vocabulary": vocabulary,
        "source_text_mask_padding": True, "target_text_mask_padding": False,
        "source_text_audio_only": True, "target_text_audio_only": False,
        "intentional_runtime_changes": ["AT to AVT", "restore original fixed-13 soft Gaussian prior",
                                        "retain downstream text_mask_padding=False and checkpoint_activations=False"],
        "source_ctc_lambda": contract["model"]["ctc_lambda"],
        "visual_weights_pretrained": False, "ctc_weights_pretrained": False,
        "speaker_weights_pretrained": True, "optimizer_restored": False, "downstream_initial_update": 0,
        "ema_policy": "Carry parent EMA step 50000, matching the original fixed-13 initialization policy",
        "function_preserving": False,
    }
