"""Prepare immutable GRID training caches from complete MPEG audio and mouth frames.

The original train/val lists define the split. Each extraction stage is resumable
through per-utterance receipts binding the source, model, protocol and output
hashes. Only a successful full audit publishes ``complete.json``. A limited run
is explicitly marked as a smoke cache and cannot become a full training cache.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from aligndit.script.eval.prepare_grid_reference import (
    atomic_bytes, canonical_json, jsonl_bytes, posterior_stats, stable_reference_seed, write_json,
)
from aligndit.script.eval.semantic_vae_decoder import (
    HOP_LENGTH, LATENT_DIM, SAMPLE_RATE, load_semantic_vae, read_json_object, sha256_file,
)


NORMALIZATION_SHA256 = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
VOCAB_SHA256 = "225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d"
VAE_SHA256 = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
AVHUBERT_SHA256 = "343c9dbaa29847b8c4d0d9503c43f2877fd2efde9988d41abf14e74f28232b75"
WAVLM_CONFIG_SHA256 = "fea6df1c2700a3954fc07e70588aecc9055eeb28db2ff57151a2db0d19180ed4"
WAVLM_PREPROCESSOR_SHA256 = "99272fe8ccfab114b68b478681ea47ee3a1ce62bb788cb92dd6e4f69fb1f1da2"
STAGES = ("audio", "latent", "video", "speaker", "repa")


def digest_object(value: dict) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def checked_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {path}: expected={expected}, got={actual}")
    return actual


def implementation_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    return {relative: sha256_file(package / relative) for relative in (
        "script/misc/prepare_grid.py", "script/misc/grid_sources.py",
        "script/eval/prepare_grid_reference.py", "script/eval/semantic_vae_decoder.py",
        "script/misc/extract_campplus_speaker_embeddings.py",
        "script/misc/extract_wavlm_base_plus_repa.py", "script/misc/avhubert_utils.py",
    )}


def pinned_assets(args) -> tuple[dict, dict]:
    """Resolve local weights without downloading or loading GPU models."""
    from huggingface_hub import hf_hub_download
    from aligndit.model.repa import (
        WAVLM_BASE_PLUS_CHECKPOINT_SHA256, WAVLM_BASE_PLUS_MODEL_ID, WAVLM_BASE_PLUS_REVISION,
    )
    from aligndit.script.misc.extract_campplus_speaker_embeddings import EXPECTED_CHECKPOINT_SHA256, MODEL_ID

    spec = read_json_object(args.semantic_vae_spec)
    if spec.get("extraction", {}).get("protocol") != "semantic_vae_posterior_sample_v1":
        raise ValueError("Semantic-VAE source spec must describe fixed posterior samples")
    if spec.get("checkpoint", {}).get("ema_sha256") != VAE_SHA256:
        raise ValueError("Semantic-VAE source spec does not describe the pinned 1000k EMA")
    checked_hash(args.normalization, NORMALIZATION_SHA256, "LibriSpeech normalization")
    checked_hash(args.vocab, VOCAB_SHA256, "Inherited vocabulary")
    checked_hash(args.semantic_vae_checkpoint / "dac/ema_state_dict.pth", VAE_SHA256, "Semantic-VAE")
    checked_hash(args.avhubert_checkpoint, AVHUBERT_SHA256, "AV-HuBERT")
    checked_hash(args.speaker_checkpoint, EXPECTED_CHECKPOINT_SHA256, "CAM++")
    teacher_files = {}
    teacher_names = ("config.json", "preprocessor_config.json", "pytorch_model.bin")
    local_complete = all((args.wavlm_model_dir / name).is_file() for name in teacher_names)
    for filename in teacher_names:
        local_file = args.wavlm_model_dir / filename
        if local_complete:
            path = local_file
        else:
            path = Path(hf_hub_download(
                repo_id=WAVLM_BASE_PLUS_MODEL_ID, revision=WAVLM_BASE_PLUS_REVISION,
                filename=filename, local_files_only=not args.download_missing_models,
                local_dir=str(args.wavlm_model_dir) if args.download_missing_models else None,
            ))
        if filename == "pytorch_model.bin":
            args.wavlm_resolved_dir = path.parent
        teacher_files[filename] = sha256_file(path)
    if teacher_files != {
        "config.json": WAVLM_CONFIG_SHA256, "preprocessor_config.json": WAVLM_PREPROCESSOR_SHA256,
        "pytorch_model.bin": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    }:
        raise RuntimeError("Local WavLM-Base+ weight/config/preprocessor differs from the pinned teacher")
    contract = {
        "latent": {"dimension": 64, "dtype": "float32", "frame_rate_hz": 40.0,
                   "feature": "semantic_vae_posterior_sample_v1", "checkpoint_sha256": VAE_SHA256},
        "video": {"dimension": 1024, "dtype": "float32", "frame_rate_hz": 40.0,
                  "feature": "avhubert_video_25hz_to_40hz_linear_align_corners_false_v1",
                  "checkpoint_sha256": AVHUBERT_SHA256},
        "speaker": {"dimension": 192, "dtype": "float32", "model_id": MODEL_ID,
                    "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
                    "source_audio": "complete_unmasked_waveform"},
        "repa": {"dimension": 768, "dtype": "float16", "model_id": WAVLM_BASE_PLUS_MODEL_ID,
                 "model_revision": WAVLM_BASE_PLUS_REVISION,
                 "checkpoint_sha256": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
                 "teacher_layer": 12, "source_audio": "complete_unmasked_waveform"},
    }
    return spec, {"feature_contract": contract, "wavlm_files": teacher_files}


def reuse_plan(args) -> tuple[list[dict], dict, dict] | None:
    """Validate frozen plan/asset bindings without repeating corpus decoding."""
    path = args.output_dir / "manifests/inventory_meta.json"
    if not path.is_file():
        return None
    spec, assets = pinned_assets(args)
    metadata = read_json_object(path)
    selection = {"mode": "full"} if args.limit is None else {"mode": "smoke", "limit": args.limit}
    if (metadata.get("schema_version") != 1 or metadata.get("dataset") != "GRID"
            or metadata.get("selection") != selection
            or metadata.get("feature_contract") != assets["feature_contract"]
            or metadata.get("normalization_sha256") != NORMALIZATION_SHA256
            or metadata.get("vocab_sha256") != VOCAB_SHA256
            or metadata.get("source_summary", {}).get("dataset_root") != str(args.grid_root.resolve())):
        raise RuntimeError("Existing GRID plan differs from requested dataset/selection/assets; choose a new output")
    protocol = metadata.get("extraction_protocol", {})
    if (protocol.get("version") != "grid_full_mpeg_audio_gray96_training_v1"
            or protocol.get("device_type") != torch.device(args.device).type
            or protocol.get("torch") != torch.__version__ or protocol.get("numpy") != np.__version__
            or protocol.get("semantic_vae_source_spec_sha256") != sha256_file(args.semantic_vae_spec)
            or protocol.get("implementation_sha256") != implementation_hashes()
            or protocol.get("wavlm_files") != assets["wavlm_files"]
            or metadata.get("extraction_sha256") != digest_object(protocol)):
        raise RuntimeError("Existing GRID extraction protocol/runtime differs; choose a new output")
    from aligndit.script.misc.grid_sources import SPLIT_SUBDIR
    for split in ("train", "val"):
        source_hash = sha256_file(args.grid_root / SPLIT_SUBDIR / f"{split}.txt")
        if metadata["source_summary"]["splits"][split]["sha256"] != source_hash:
            raise RuntimeError(f"GRID {split} source list changed since plan creation")
    if args.source_audit_dir is not None:
        actual = {name: sha256_file(args.source_audit_dir / name) for name in (
            "source_records.jsonl", "source_exclusions.jsonl", "source_summary.json",
        )}
        if actual != metadata.get("source_audit_sha256"):
            raise RuntimeError("Existing GRID plan is bound to different source audit files")
    for name, entry in metadata["manifests"].items():
        manifest = args.output_dir / "manifests" / name
        if manifest.stat().st_size != entry["size_bytes"] or sha256_file(manifest) != entry["sha256"]:
            raise RuntimeError(f"Frozen GRID manifest was modified: {manifest}")
    rows = [json.loads(line) for line in (args.output_dir / "manifests/inventory.jsonl").read_text().splitlines()]
    if (len(rows) != metadata["manifests"]["inventory.jsonl"]["count"]
            or dict(Counter(row["split"] for row in rows)) != metadata["split_counts"]):
        raise RuntimeError("Frozen GRID inventory count mismatch")
    return rows, metadata, spec


def build_plan(args) -> tuple[list[dict], dict, dict]:
    from aligndit.script.misc.grid_sources import SPLIT_SUBDIR, build_source_records

    spec, assets = pinned_assets(args)
    audit_hashes = None
    if args.source_audit_dir is None:
        sources, exclusions, summary = build_source_records(
            args.grid_root, limit=args.limit, workers=args.workers,
            ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
        )
    else:
        audit = args.source_audit_dir.resolve(strict=True)
        def read_rows(name):
            return [json.loads(line) for line in (audit / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        sources = read_rows("source_records.jsonl")
        exclusions = read_rows("source_exclusions.jsonl")
        summary = read_json_object(audit / "source_summary.json")
        if (summary.get("schema_version") != 1 or summary.get("dataset_root") != str(args.grid_root.resolve())
                or summary.get("video_decode_validated") is not True
                or summary.get("limit_per_split") != args.limit):
            raise ValueError("Source audit root/selection/decoded-video contract differs from this request")
        for split in ("train", "val"):
            split_hash = sha256_file(args.grid_root / SPLIT_SUBDIR / f"{split}.txt")
            if summary.get("splits", {}).get(split, {}).get("sha256") != split_hash:
                raise RuntimeError(f"GRID {split} list changed after the source audit")
            if any(row.get("source_split_sha256") != split_hash for row in sources if row.get("split") == split):
                raise ValueError(f"Source audit records do not match the {split} split fingerprint")
        if (dict(Counter(row["split"] for row in sources)) != summary.get("accepted")
                or dict(Counter(row["split"] for row in exclusions)) != summary.get("excluded")):
            raise ValueError("Source audit summary counts do not match its records")
        if any(row.get("video_validation") != "decoded_frame_count" for row in sources):
            raise ValueError("Source audit contains records without video decode validation")
        audit_hashes = {name: sha256_file(audit / name) for name in (
            "source_records.jsonl", "source_exclusions.jsonl", "source_summary.json",
        )}
    characters = args.vocab.read_text(encoding="utf-8").splitlines()
    vocabulary = {character: index for index, character in enumerate(characters)}
    if vocabulary.get(" ") != 0 or len(vocabulary) != len(characters):
        raise ValueError("Inherited vocabulary must have unique entries and space at index zero")
    rows = []
    for source in sources:
        text = source["text"]
        unknown = set(text) - set(vocabulary)
        if unknown:
            raise ValueError(f"GRID text has out-of-vocabulary characters: {source['id']}: {unknown}")
        ids = [vocabulary[character] for character in text]
        repeats = sum(left == right for left, right in zip(ids, ids[1:]))
        frames = math.ceil(source["num_samples"] / HOP_LENGTH)
        relative = Path(source["split"]) / source["speaker"] / source["utterance_id"]
        key = "grid/" + source["id"]
        rows.append({
            "utterance_key": key, "split": source["split"], "speaker": source["speaker"],
            "text": text, "duration_seconds": source["num_samples"] / SAMPLE_RATE,
            "original_num_samples": source["num_samples"], "sample_rate": SAMPLE_RATE,
            "latent_frames": frames, "latent_dim": LATENT_DIM, "video_dim": 1024,
            "source_video_frames": source["frame_count"],
            "audio_relative_path": str(relative.with_suffix(".wav")),
            "latent_relative_path": str(Path("latents") / relative.with_suffix(".npy")),
            "video_40hz_relative_path": str(Path("video_40hz") / relative.with_suffix(".npy")),
            "ctc_target_length": len(ids), "ctc_adjacent_repeats": repeats,
            "ctc_min_input_frames": len(ids) + repeats,
            "ctc_feasible_40hz": frames >= len(ids) + repeats,
            "posterior_seed": stable_reference_seed(key),
            "source": source, "source_sha256": digest_object(source),
        })
    if not rows or len({row["utterance_key"] for row in rows}) != len(rows):
        raise ValueError("GRID inventory is empty or has duplicate utterances")
    split_counts = dict(Counter(row["split"] for row in rows))
    if set(split_counts) != {"train", "val"}:
        raise ValueError("GRID preparation requires nonempty train and val splits")
    manifests = {}
    for split in ("inventory", "train", "val"):
        selected = rows if split == "inventory" else [row for row in rows if row["split"] == split]
        payload = jsonl_bytes(selected)
        manifests[f"{split}.jsonl"] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "count": len(selected), "size_bytes": len(payload),
        }
    protocol = {
        "version": "grid_full_mpeg_audio_gray96_training_v1",
        "canonical_audio": "16000Hz_mono_PCM16_full_MPEG_audio_aligned_to_first_video_frame",
        "audio_length": "decoded_video_frame_count_times_640_samples",
        "video_input": "ordered_existing_96x96_grayscale_mouth_frames",
        "video_transform": "AVHubert_checkpoint_normalization_center_crop_no_augmentation",
        "video_interpolation": "linear_align_corners_false_to_exact_latent_frames",
        "posterior_base_seed": 666, "posterior_logvar_clamp": [-12, 12],
        "posterior_noise": "torch.randn_per_utterance_device_Generator",
        "posterior_padding": "right_zero_pad_to_400_samples", "latent_normalized": False,
        "speaker_padding": "repeat_complete_utterance_to_10_second_chunk_multiple",
        "speaker_aggregation": "arithmetic_mean_then_l2_normalize",
        "repa": "full_audio_last_hidden_state_native_50Hz_no_interpolation",
        "device_type": torch.device(args.device).type, "torch": torch.__version__,
        "implementation_sha256": implementation_hashes(),
        "numpy": np.__version__, "semantic_vae_source_spec_sha256": sha256_file(args.semantic_vae_spec),
        "normalization_sha256": NORMALIZATION_SHA256, "vocab_sha256": VOCAB_SHA256,
        "feature_contract": assets["feature_contract"], "wavlm_files": assets["wavlm_files"],
    }
    metadata = {
        "schema_version": 1, "dataset": "GRID",
        "selection": {"mode": "full"} if args.limit is None else {"mode": "smoke", "limit": args.limit},
        "manifests": manifests, "split_counts": split_counts,
        "normalization_sha256": NORMALIZATION_SHA256, "vocab_sha256": VOCAB_SHA256,
        "feature_contract": assets["feature_contract"], "extraction_protocol": protocol,
        "extraction_sha256": digest_object(protocol),
        "source_summary": summary, "exclusions": exclusions,
        "source_audit_sha256": audit_hashes,
        "normalization_path": str(args.normalization.resolve()), "vocab_path": str(args.vocab.resolve()),
    }
    return rows, metadata, spec


def artifact_path(output: Path, row: dict, stage: str) -> Path:
    relative = Path(row["audio_relative_path"])
    if stage == "audio":
        return output / "audio" / relative
    if stage == "latent":
        return output / row["latent_relative_path"]
    if stage == "video":
        return output / row["video_40hz_relative_path"]
    return output / ("speakers" if stage == "speaker" else "repa") / relative.with_suffix(".npy")


def receipt_path(output: Path, row: dict, stage: str) -> Path:
    return output / "receipts" / stage / Path(row["audio_relative_path"]).with_suffix(".json")


def validate_artifact(path: Path, row: dict, stage: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Missing or symlinked {stage} artifact: {path}")
    if stage == "audio":
        value, rate = sf.read(path, dtype="float32", always_2d=True)
        info = sf.info(path)
        if rate != SAMPLE_RATE or value.shape != (row["original_num_samples"], 1) or info.subtype != "PCM_16":
            raise ValueError(f"Invalid canonical GRID audio: {path}")
    else:
        value = np.load(path, allow_pickle=False)
        if stage == "speaker":
            from aligndit.model.speaker_embedding import validate_speaker_embedding_array
            validate_speaker_embedding_array(value, source=path)
        elif stage == "repa":
            from aligndit.model.repa import validate_repa_feature_array
            validate_repa_feature_array(value, source=path)
            # WavLM convolution stride 320, receptive field 400 samples.
            expected = (row["original_num_samples"] - 400) // 320 + 1
            if value.shape[0] != expected:
                raise ValueError(f"WavLM duration mismatch: {path}: {value.shape[0]} != {expected}")
        else:
            dimension = LATENT_DIM if stage == "latent" else 1024
            if value.shape != (row["latent_frames"], dimension) or value.dtype != np.float32:
                raise ValueError(f"Invalid {stage} shape/dtype: {path}: {value.shape}, {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite {stage} artifact: {path}")


def validate_receipt(output: Path, row: dict, stage: str, fingerprint: str) -> bool:
    path = receipt_path(output, row, stage)
    if not path.exists():
        return False
    receipt = read_json_object(path)
    expected = {"utterance_key": row["utterance_key"], "source_sha256": row["source_sha256"],
                "extraction_sha256": fingerprint, "stage": stage}
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Stale GRID cache receipt; use a new cache directory: {path}")
    artifact = artifact_path(output, row, stage)
    validate_artifact(artifact, row, stage)
    if sha256_file(artifact) != receipt.get("artifact_sha256"):
        raise RuntimeError(f"GRID cache artifact hash mismatch: {artifact}")
    if stage in {"latent", "speaker", "repa"}:
        audio_receipt = read_json_object(receipt_path(output, row, "audio"))
        if receipt.get("audio_sha256") != audio_receipt.get("artifact_sha256"):
            raise RuntimeError(f"GRID feature is bound to different canonical audio: {artifact}")
    return True


def publish_receipt(output: Path, row: dict, stage: str, fingerprint: str) -> None:
    path = artifact_path(output, row, stage)
    validate_artifact(path, row, stage)
    receipt = {"utterance_key": row["utterance_key"], "source_sha256": row["source_sha256"],
               "extraction_sha256": fingerprint, "stage": stage, "artifact_sha256": sha256_file(path)}
    if stage in {"latent", "speaker", "repa"}:
        receipt["audio_sha256"] = sha256_file(artifact_path(output, row, "audio"))
    write_json(receipt_path(output, row, stage), receipt)


def save_array(path: Path, value: np.ndarray) -> None:
    from aligndit.script.misc.extract_campplus_speaker_embeddings import atomic_save_npy
    atomic_save_npy(path, value)


def extract_audio(output: Path, row: dict, args) -> None:
    from aligndit.script.misc.grid_sources import decode_aligned_audio
    checked_hash(Path(row["source"]["video_path"]), row["source"]["video_sha256"], "GRID source MPG")
    waveform = decode_aligned_audio(row["source"], ffmpeg=args.ffmpeg)
    path = artifact_path(output, row, "audio")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".wav", dir=path.parent)
    os.close(descriptor)
    try:
        sf.write(temporary, waveform, SAMPLE_RATE, subtype="PCM_16")
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_avhubert(args, device):
    if str(args.fairseq_repo) not in sys.path:
        sys.path.insert(0, str(args.fairseq_repo))
    if str(args.avhubert_user_dir) not in sys.path:
        sys.path.insert(0, str(args.avhubert_user_dir))
    import fairseq
    from aligndit.script.misc import avhubert_utils
    # The bundled AV-HuBERT enables a different, non-package debug import path
    # when len(sys.argv)==1. Force ordinary package imports even for defaults.
    original_argv = sys.argv
    try:
        if len(sys.argv) == 1:
            sys.argv = [*sys.argv, "--grid-feature-extraction"]
        fairseq.utils.import_user_module(SimpleNamespace(user_dir=str(args.avhubert_user_dir)))
    finally:
        sys.argv = original_argv
    models, _, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([str(args.avhubert_checkpoint)])
    model = models[0].eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    transform = avhubert_utils.Compose([
        avhubert_utils.Normalize(0.0, 255.0),
        avhubert_utils.CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        avhubert_utils.Normalize(task.cfg.image_mean, task.cfg.image_std),
    ])
    return model, transform


@torch.inference_mode()
def run_stage(stage: str, rows: list[dict], metadata: dict, spec: dict, args) -> None:
    output = args.output_dir
    fingerprint = metadata["extraction_sha256"]
    pending = [row for row in rows if not validate_receipt(output, row, stage, fingerprint)]
    print(f"GRID {stage}: verified={len(rows) - len(pending)}, pending={len(pending)}", flush=True)
    if not pending:
        return
    device = torch.device(args.device)
    model = None
    if stage in {"latent", "speaker", "repa"}:
        for row in pending:
            if not validate_receipt(output, row, "audio", fingerprint):
                raise RuntimeError("Run --stage audio before extracting audio features")
    if stage == "latent":
        model, _ = load_semantic_vae(repo=args.semantic_vae_repo,
                                    checkpoint_root=args.semantic_vae_checkpoint,
                                    cache_spec=spec, device=torch.device("cpu"))
        del model.decoder
        model.to(device=device, dtype=torch.float32)
        posterior_stats(model, torch.zeros(1, 1, HOP_LENGTH, device=device))
    elif stage == "video":
        model, transform = load_avhubert(args, device)
    elif stage == "speaker":
        from aligndit.script.misc.extract_campplus_speaker_embeddings import (
            AudioFeatureDataset, ExtractionItem, load_campplus,
        )
        model = load_campplus(args.speaker_checkpoint, device)
    elif stage == "repa":
        from aligndit.script.misc.extract_wavlm_base_plus_repa import extract_batch, ExtractionItem
        from transformers import AutoFeatureExtractor, WavLMModel
        feature_extractor = AutoFeatureExtractor.from_pretrained(str(args.wavlm_resolved_dir), local_files_only=True)
        model = WavLMModel.from_pretrained(str(args.wavlm_resolved_dir), local_files_only=True,
                                          torch_dtype=torch.float32)
        if model.config.hidden_size != 768 or model.config.num_hidden_layers != 12:
            raise RuntimeError("WavLM architecture differs from the pinned 12-layer 768-D Base+ teacher")
        model = model.eval().requires_grad_(False).to(device)
    try:
        for row in tqdm(pending, desc=f"GRID {stage}", unit="utterance"):
            path = artifact_path(output, row, stage)
            if stage == "audio":
                extract_audio(output, row, args)
            elif stage == "latent":
                waveform, _ = sf.read(artifact_path(output, row, "audio"), dtype="float32")
                waveform = torch.from_numpy(waveform).to(device).view(1, 1, -1)
                waveform = F.pad(waveform, (0, row["latent_frames"] * HOP_LENGTH - waveform.shape[-1]))
                mu, logvar = posterior_stats(model, waveform)
                generator = torch.Generator(device=device).manual_seed(row["posterior_seed"])
                noise = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
                value = (mu + torch.exp(0.5 * logvar) * noise).squeeze(0).cpu().numpy().astype(np.float32)
                save_array(path, value)
            elif stage == "video":
                from aligndit.script.misc.grid_sources import _mouth_digest, load_mouth_frames, mouth_frame_paths
                if _mouth_digest(mouth_frame_paths(row["source"]["mouth_dir"])) != row["source"]["mouth_sha256"]:
                    raise RuntimeError(f"GRID mouth frames changed after audit: {row['utterance_key']}")
                frames = transform(load_mouth_frames(row["source"]))
                video = torch.from_numpy(frames.astype(np.float32)).to(device)[None, None]
                feature, _ = model.extract_finetune(source={"audio": None, "video": video},
                                                    padding_mask=None, mask=False)
                if feature.shape != (1, row["source_video_frames"], 1024):
                    raise ValueError(f"Unexpected AV-HuBERT output geometry: {feature.shape}")
                feature = F.interpolate(feature.transpose(1, 2), size=row["latent_frames"],
                                        mode="linear", align_corners=False).transpose(1, 2)
                save_array(path, feature.squeeze(0).float().cpu().numpy())
            elif stage == "speaker":
                item = ExtractionItem(str(artifact_path(output, row, "audio")), str(path), row["split"])
                _, features, error = AudioFeatureDataset([item])[0]
                if error:
                    raise RuntimeError(error)
                embedding = model(features.to(device=device, dtype=torch.float32)).float().mean(dim=0)
                save_array(path, F.normalize(embedding, dim=0).cpu().numpy().astype(np.float32))
            else:
                item = ExtractionItem(row["utterance_key"], artifact_path(output, row, "audio"), path,
                                      row["duration_seconds"])
                extract_batch([item], feature_extractor, model, device, 90.0)
            publish_receipt(output, row, stage, fingerprint)
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def verify(rows: list[dict], metadata: dict, args, *, required: bool) -> bool:
    missing, invalid = [], []
    for row in tqdm(rows, desc="Verifying all GRID features", unit="utterance"):
        for stage in STAGES:
            try:
                if not validate_receipt(args.output_dir, row, stage, metadata["extraction_sha256"]):
                    missing.append(str(artifact_path(args.output_dir, row, stage)))
            except Exception as error:
                invalid.append({"utterance_key": row["utterance_key"], "stage": stage,
                                "error": f"{type(error).__name__}: {error}"})
    report = {
        "schema_version": 1, "dataset": "GRID", "complete": not missing and not invalid,
        "inventory_meta_sha256": sha256_file(args.output_dir / "manifests/inventory_meta.json"),
        "inventory_sha256": metadata["manifests"]["inventory.jsonl"]["sha256"],
        "count": len(rows), "split_counts": metadata["split_counts"],
        "feature_contract": metadata["feature_contract"], "missing": missing, "invalid": invalid,
        "selection": metadata["selection"], "extraction_sha256": metadata["extraction_sha256"],
    }
    write_json(args.output_dir / "coverage_report.json", report, replace=True)
    if report["complete"]:
        write_json(args.output_dir / "complete.json", report, replace=True)
        print(f"GRID cache complete: {metadata['split_counts']} at {args.output_dir}", flush=True)
    else:
        (args.output_dir / "complete.json").unlink(missing_ok=True)
        if required:
            raise RuntimeError(f"GRID cache incomplete: missing={len(missing)}, invalid={len(invalid)}; "
                               f"inspect {args.output_dir / 'coverage_report.json'}")
    return report["complete"]


def prepare(args) -> None:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive and applies independently to train and val")
    if args.workers <= 0 or args.torch_threads <= 0:
        raise ValueError("Worker and torch thread counts must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Use one preparation process per cache; distributed extraction is not supported")
    args.output_dir = args.output_dir.expanduser().resolve()
    repository = Path(__file__).resolve().parents[5]
    if args.output_dir.is_relative_to(repository):
        raise ValueError("GRID runtime caches must be outside my_papers_code")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.torch_threads)
    with (args.output_dir / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = reuse_plan(args)
        rows, metadata, spec = previous if previous is not None else build_plan(args)
        # Plans and all source fingerprints are immutable, including split lists.
        for split in ("inventory", "train", "val"):
            selected = rows if split == "inventory" else [row for row in rows if row["split"] == split]
            atomic_bytes(args.output_dir / f"manifests/{split}.jsonl", jsonl_bytes(selected))
        write_json(args.output_dir / "manifests/inventory_meta.json", metadata)
        atomic_bytes(args.output_dir / "state/latents/spec.json", args.semantic_vae_spec.read_bytes())
        atomic_bytes(args.output_dir / "state/train_normalization.json", args.normalization.read_bytes())
        atomic_bytes(args.output_dir / "vocab.txt", args.vocab.read_bytes())
        print(f"Frozen GRID plan: {metadata['split_counts']}; selection={metadata['selection']}", flush=True)
        if args.plan_only:
            return
        marker = args.output_dir / "WRITE_ACTIVE.json"
        write_json(marker, {"pid": os.getpid(), "stage": "verify" if args.verify_only else args.stage}, replace=True)
        (args.output_dir / "complete.json").unlink(missing_ok=True)
        try:
            if not args.verify_only:
                device = torch.device(args.device)
                if device.type == "cuda":
                    if not torch.cuda.is_available():
                        raise RuntimeError("CUDA is unavailable; --plan-only needs no GPU")
                    torch.cuda.set_device(device)
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = False
                torch.use_deterministic_algorithms(False)
                for stage in STAGES if args.stage == "all" else (args.stage,):
                    run_stage(stage, rows, metadata, spec, args)
            verify(rows, metadata, args, required=args.verify_only or args.stage == "all")
        finally:
            marker.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    prefix = os.environ.get("ROOT_PREFIX", "")
    def local(relative: str) -> Path:
        return Path(f"{prefix}/zjw524/{relative}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=local("projects/data/GRID_svae1000k_sample_seed666_fp32"))
    parser.add_argument("--grid-root", type=Path, default=local("datasets/Grid_Dataset"))
    parser.add_argument("--source-audit-dir", type=Path,
                        help="Reuse grid_sources.py audit JSONL/summary with identical root/splits/limit")
    parser.add_argument("--limit", type=int, help="Smoke selection: first N eligible records per original split")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--ffmpeg", default=str(Path(sys.executable).parent / "ffmpeg"))
    parser.add_argument("--ffprobe", default=str(Path(sys.executable).parent / "ffprobe"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--plan-only", action="store_true")
    modes.add_argument("--verify-only", action="store_true")
    parser.add_argument("--stage", choices=("all",) + STAGES, default="all")
    parser.add_argument("--semantic-vae-repo", type=Path, default=local("projects/alignDiT_idea6/papers_codes/Semantic-VAE"))
    parser.add_argument("--semantic-vae-checkpoint", type=Path, default=local("projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k"))
    parser.add_argument("--semantic-vae-spec", type=Path, default=local("projects/data/CelebVDub_svae1000k_sample_seed666_fp32/state/latents/spec.json"))
    parser.add_argument("--normalization", type=Path, default=local("projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json"))
    parser.add_argument("--vocab", type=Path, default=local("projects/data/CelebVDub_char/vocab.txt"))
    parser.add_argument("--speaker-checkpoint", type=Path, default=local("projects/data/pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"))
    parser.add_argument("--wavlm-model-dir", type=Path, default=local("projects/data/pretrained_models/wavlm-base-plus"))
    parser.add_argument("--download-missing-models", action="store_true",
                        help="Allow download of WavLM files from the exact pinned Hub revision if absent locally")
    parser.add_argument("--avhubert-checkpoint", type=Path, default=local("projects/data/large_vox_iter5.pt"))
    parser.add_argument("--fairseq-repo", type=Path, default=local("projects/data/av_hubert/fairseq/fairseq"))
    parser.add_argument("--avhubert-user-dir", type=Path, default=local("projects/data/av_hubert/avhubert/avhubert"))
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
