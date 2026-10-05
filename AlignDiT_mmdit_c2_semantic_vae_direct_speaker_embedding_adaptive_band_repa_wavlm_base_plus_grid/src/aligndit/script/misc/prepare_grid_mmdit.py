"""Build immutable GRID Semantic-VAE / video / CAM++ / WavLM training caches.

The input is the independently audited AlignDiT GRID baseline cache. Its audio
contains the complete, aligned MPG soundtrack; the distributed GRID WAVs are
silence-trimmed and must not replace it. This script never changes source data.
Use a separate output directory with --limit-per-split for smoke validation.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from tqdm import tqdm

from aligndit.model.repa import (
    WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
    WAVLM_BASE_PLUS_MODEL_ID,
    WAVLM_BASE_PLUS_REVISION,
    validate_repa_feature_array,
)
from aligndit.model.speaker_embedding import validate_speaker_embedding_array
from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae, sha256_file
from aligndit.script.misc.extract_campplus_speaker_embeddings import (
    EXPECTED_CHECKPOINT_SHA256,
    MODEL_ID,
    AudioFeatureDataset,
    ExtractionItem,
    atomic_save_json,
    atomic_save_npy,
    load_campplus,
)


NORMALIZATION_SHA = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
VOCAB_SHA = "225df7792c4ade59e3de39789b36fdf735e1b30ed96b4456d2d27df0d86a875d"
FULL_COUNTS = {"train": 29557, "val": 3281}
FEATURE_KEYS = ("latent", "video_40hz", "speaker", "repa")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != value:
            raise RuntimeError(f"Preparation contract changed; use a new output directory: {path}")
    else:
        atomic_save_json(path, value)


def write_jsonl(path, rows):
    contents = "".join(canonical(row) + "\n" for row in rows)
    if path.exists():
        if path.read_text(encoding="utf-8") != contents:
            raise RuntimeError(f"Manifest changed; use a new output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text(contents, encoding="utf-8")
    temporary.replace(path)


def read_records(root):
    spec = read_json(root / "spec.json")
    path = root / "manifests/inventory.jsonl"
    if sha256_file(path) != spec["manifests"]["inventory.jsonl"]["sha256"]:
        raise RuntimeError("GRID inventory differs from frozen preparation spec")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def select_records(rows, limit):
    if not limit:
        return rows
    # Exercise the 74-frame/119-latent boundary, followed by ordinary 75/120.
    result = []
    for split in FULL_COUNTS:
        candidates = [row for row in rows if row["split"] == split]
        candidates.sort(key=lambda row: (row["frames"], row["id"]))
        short = [row for row in candidates if row["frames"] == 74]
        normal = [row for row in candidates if row["frames"] == 75]
        ordered = short[:1] + normal + short[1:]
        result.extend(ordered[:limit])
    return result


def manifest(args):
    source = args.source_root.resolve(strict=True)
    source_complete = read_json(source / "complete.json")
    source_manifest = read_json(source / "manifest.json")
    source_rows = read_json(source / "records.json")
    row_sha = hashlib.sha256(json.dumps(source_rows, sort_keys=True).encode()).hexdigest()
    if (
        source_complete.get("counts") != FULL_COUNTS
        or source_complete.get("train_val_overlap") != 0
        or source_manifest.get("rows_sha256") != row_sha
        or source_manifest.get("limit_per_split") != 0
        or source_manifest.get("audio") != "complete MPG audio; 16 kHz mono, aligned to original 25 fps video"
        or source_complete.get("feature_shape") != "T x 1024"
    ):
        raise RuntimeError("Expected the complete, audited GRID baseline cache")
    seen = set()
    for row in source_rows:
        if row["id"] in seen or row["split"] not in FULL_COUNTS or row["frames"] not in (74, 75):
            raise RuntimeError(f"Invalid or overlapping GRID source record: {row}")
        seen.add(row["id"])
    if dict(Counter(row["split"] for row in source_rows)) != FULL_COUNTS:
        raise RuntimeError("GRID source records have unexpected split counts")
    for path, expected in ((args.normalization, NORMALIZATION_SHA), (args.vocab, VOCAB_SHA)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"Fixed pretrained normalization/vocabulary mismatch: {path}")
    vocabulary = {character: index for index, character in enumerate(args.vocab.read_text().splitlines())}
    if vocabulary.get(" ") != 0:
        raise RuntimeError("Pretrained vocabulary must map space to zero")
    records = []
    for row in tqdm(select_records(source_rows, args.limit_per_split), desc="GRID manifest", unit="clip"):
        relative = Path(row["split"]) / row["speaker"] / row["utterance"]
        audio = source / "GRID/audio" / relative.with_suffix(".wav")
        video = source / "GRID/avhubert_video_feat" / relative.with_suffix(".npy")
        audio_state = read_json(source / "GRID/state/audio" / relative.with_suffix(".json"))
        video_state = read_json(source / "GRID/state/visual" / relative.with_suffix(".json"))
        audio_fp, video_fp = fingerprint(audio), fingerprint(video)
        if audio_state.get("audio") != audio_fp or video_state.get("feature") != video_fp:
            raise RuntimeError(f"Audited source feature changed: {relative}")
        if video_state.get("key", {}).get("input") != "video_only":
            raise RuntimeError(f"AV-HuBERT must have used video only: {relative}")
        samples = row["frames"] * 640
        frames = math.ceil(samples / 400)
        text = row["text"]
        if any(character not in vocabulary for character in text):
            raise RuntimeError(f"GRID transcript outside pretrained vocabulary: {text!r}")
        repeats = sum(a == b for a, b in pairwise(text))
        key = f"grid/{relative.as_posix()}"
        seed = int.from_bytes(hashlib.sha256(f"666:{key}".encode()).digest()[:8], "little") % (2**63 - 1)
        records.append(
            {
                "utterance_key": key,
                "sample_id": row["id"],
                "split": row["split"],
                "speaker_id": row["speaker"],
                "text": text,
                "audio_relative_path": str(relative.with_suffix(".wav")),
                "audio_path": str(audio.resolve()),
                "video_source_path": str(video.resolve()),
                "audio_fingerprint": audio_fp,
                "video_fingerprint": video_fp,
                "original_num_samples": samples,
                "duration_seconds": samples / 16000,
                "source_video_frames": row["frames"],
                "latent_frames": frames,
                "latent_dim": 64,
                "video_dim": 1024,
                "posterior_seed": seed,
                "latent_relative_path": str(Path("latents") / relative.with_suffix(".npy")),
                "video_40hz_relative_path": str(Path("video_40hz") / relative.with_suffix(".npy")),
                "speaker_relative_path": str(Path("speaker_embeddings") / relative.with_suffix(".npy")),
                "repa_relative_path": str(Path("repa") / relative.with_suffix(".npy")),
                "ctc_target_length": len(text),
                "ctc_adjacent_repeats": repeats,
                "ctc_min_input_frames": len(text) + repeats,
                "ctc_feasible_40hz": frames >= len(text) + repeats,
            }
        )
    manifests = {}
    for name, selected in (
        ("inventory.jsonl", records),
        ("train.jsonl", [r for r in records if r["split"] == "train"]),
        ("val.jsonl", [r for r in records if r["split"] == "val"]),
    ):
        path = args.output / "manifests" / name
        write_jsonl(path, selected)
        manifests[name] = {"sha256": sha256_file(path), "count": len(selected)}
    vae_spec = read_json(args.vae_spec)
    if (
        vae_spec.get("extraction", {}).get("protocol") != "semantic_vae_posterior_sample_v1"
        or vae_spec.get("checkpoint", {}).get("ema_sha256")
        != "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
    ):
        raise RuntimeError("Expected the original Semantic-VAE 1000k EMA contract")
    spec = {
        "schema_version": 1,
        "dataset": "GRID",
        "manifests": manifests,
        "split_counts": dict(Counter(row["split"] for row in records)),
        "selection": {"mode": "subset" if args.limit_per_split else "full", "limit_per_split": args.limit_per_split},
        "normalization": {"path": str(args.normalization.resolve()), "sha256": NORMALIZATION_SHA},
        "vocab": {"path": str(args.vocab.resolve()), "sha256": VOCAB_SHA},
        "source": {
            "root": str(source),
            "manifest_sha256": sha256_file(source / "manifest.json"),
            "records_sha256": sha256_file(source / "records.json"),
            "complete_sha256": sha256_file(source / "complete.json"),
        },
        "latent_spec": {
            "dimension": 64,
            "dtype": "float32",
            "frame_rate_hz": 40.0,
            "hop_length_samples": 400,
            "mode": "fixed_posterior_sample",
            "sample_rate": 16000,
        },
        "video_spec": {
            "dimension": 1024,
            "dtype": "float32",
            "frame_rate_hz": 40.0,
            "interpolation": "linear_align_corners_false_exact_latent_length",
            "source": "AV-HuBERT_Large_video_only",
        },
        "speaker_spec": {
            "dimension": 192,
            "dtype": "float32",
            "model_id": MODEL_ID,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        },
        "repa_spec": {
            "dimension": 768,
            "dtype": "float16",
            "model_id": WAVLM_BASE_PLUS_MODEL_ID,
            "model_revision": WAVLM_BASE_PLUS_REVISION,
            "checkpoint_sha256": WAVLM_BASE_PLUS_CHECKPOINT_SHA256,
            "teacher_layer": 12,
            "frame_rate_hz": 50.0,
            "config_sha256": sha256_file(args.wavlm_dir / "config.json"),
            "preprocessor_sha256": sha256_file(args.wavlm_dir / "preprocessor_config.json"),
        },
        "source_audio": "complete_unmasked_waveform",
        "semantic_vae_contract": vae_spec,
        "extraction": {
            "posterior_base_seed": 666,
            "posterior_seed_formula": "little_endian_SHA256_first8(666:key)_mod_(2**63-1)",
            "waveform_padding": "right_zero_to_multiple_400",
            "logvar_clamp": [-12, 12],
            "speaker_padding": "repeat_complete_utterance_to_10s",
            "speaker_aggregation": "mean_then_l2_normalize",
            "wavlm_input": "whole_utterance_zero_mean_unit_variance_attention_mask",
            "dtype": "float32",
            "script_sha256": sha256_file(Path(__file__)),
            "batch_size": args.batch_size,
        },
    }
    immutable_json(args.output / "spec.json", spec)
    immutable_json(args.output / "manifests/inventory_meta.json", spec)
    print("GRID manifest ready:", spec["split_counts"], flush=True)


def state_path(root, row):
    return root / "state/clips" / Path(row["audio_relative_path"]).with_suffix(".json")


def verify_features(root, row):
    result = {}
    for key in FEATURE_KEYS:
        path = root / row[f"{key}_relative_path"]
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"Missing regular GRID cache: {path}")
        array = np.load(path, allow_pickle=False)
        if key == "speaker":
            validate_speaker_embedding_array(array, source=path)
        elif key == "repa":
            validate_repa_feature_array(array, source=path)
            expected = (row["original_num_samples"] - 400) // 320 + 1
            if array.shape[0] != expected:
                raise RuntimeError(f"Wrong WavLM timeline: {path}")
        elif (
            array.shape != (row["latent_frames"], 64 if key == "latent" else 1024)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise RuntimeError(f"Invalid GRID feature: {path}")
        result[key] = sha256_file(path)
    return result


def verified_state(root, row, spec_sha):
    path = state_path(root, row)
    if not path.exists():
        return False
    state = read_json(path)
    if state.get("spec_sha256") != spec_sha or state.get("utterance_key") != row["utterance_key"]:
        raise RuntimeError(f"GRID clip extraction contract changed: {path}")
    if (
        fingerprint(row["audio_path"]) != row["audio_fingerprint"]
        or fingerprint(row["video_source_path"]) != row["video_fingerprint"]
    ):
        raise RuntimeError(f"GRID source changed: {row['utterance_key']}")
    if state.get("features") != verify_features(root, row):
        raise RuntimeError(f"GRID cached feature checksum mismatch: {path}")
    return True


class InputDataset(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if (
            fingerprint(row["audio_path"]) != row["audio_fingerprint"]
            or fingerprint(row["video_source_path"]) != row["video_fingerprint"]
        ):
            raise RuntimeError(f"GRID source changed: {row['utterance_key']}")
        waveform, rate = sf.read(row["audio_path"], dtype="float32")
        if rate != 16000 or waveform.shape != (row["original_num_samples"],) or not np.isfinite(waveform).all():
            raise RuntimeError(f"Wrong full-length GRID waveform: {row['audio_path']}")
        video = np.load(row["video_source_path"], allow_pickle=False)
        if (
            video.shape != (row["source_video_frames"], 1024)
            or video.dtype != np.float32
            or not np.isfinite(video).all()
        ):
            raise RuntimeError(f"Invalid AV-HuBERT source: {row['video_source_path']}")
        item = ExtractionItem(row["audio_path"], row["speaker_relative_path"], row["split"])
        _, features, error = AudioFeatureDataset([item])[0]
        if error:
            raise RuntimeError(f"CAM++ input failed: {error}")
        return row, waveform, features, video


def identity_collate(items):
    return items


@torch.inference_mode()
def extract(args):
    from transformers import AutoFeatureExtractor, WavLMModel

    root = args.output
    spec = read_json(root / "spec.json")
    spec_sha = sha256_file(root / "spec.json")
    if spec["extraction"]["script_sha256"] != sha256_file(Path(__file__)):
        raise RuntimeError("Preparation code changed after the manifest was frozen; use a new output directory")
    if spec["extraction"]["batch_size"] != args.batch_size:
        raise RuntimeError("Extraction batch-size must match the frozen manifest")
    pending = [
        row for row in tqdm(read_records(root), desc="Resume verification") if not verified_state(root, row, spec_sha)
    ]
    if not pending:
        print("All GRID features already verified", flush=True)
        return
    if sha256_file(args.speaker_checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError("CAM++ checkpoint differs from the speaker model contract")
    if sha256_file(args.wavlm_dir / "pytorch_model.bin") != WAVLM_BASE_PLUS_CHECKPOINT_SHA256:
        raise RuntimeError("WavLM checkpoint differs from the pinned Base+ teacher")
    for filename, key in (("config.json", "config_sha256"), ("preprocessor_config.json", "preprocessor_sha256")):
        if sha256_file(args.wavlm_dir / filename) != spec["repa_spec"][key]:
            raise RuntimeError(f"WavLM configuration changed: {filename}")
    device = torch.device(args.device)
    vae, vae_metadata = load_semantic_vae(
        repo=args.vae_repo,
        checkpoint_root=args.vae_checkpoint,
        cache_spec=spec["semantic_vae_contract"],
        device=device,
    )
    # The decoder is needed only for synthesis, not cache extraction.
    del vae.decoder
    campplus = load_campplus(args.speaker_checkpoint, device)
    extractor = AutoFeatureExtractor.from_pretrained(str(args.wavlm_dir), local_files_only=True)
    teacher = WavLMModel.from_pretrained(str(args.wavlm_dir), local_files_only=True, torch_dtype=torch.float32)
    if teacher.config.hidden_size != 768 or teacher.config.num_hidden_layers != 12:
        raise RuntimeError("WavLM teacher architecture is not Base+")
    teacher = teacher.eval().requires_grad_(False).to(device)
    runtime = {
        "device": str(device),
        "torch": torch.__version__,
        "batch_size": args.batch_size,
        "vae": vae_metadata,
        "speaker_checkpoint": str(args.speaker_checkpoint.resolve()),
        "wavlm_directory": str(args.wavlm_dir.resolve()),
    }
    atomic_save_json(root / "state/extraction_runtime.json", runtime)
    # Batch by the original waveform length: padding must never alter a VAE
    # posterior or the WavLM GroupNorm statistics for shorter utterances.
    ordered = sorted(pending, key=lambda row: (row["original_num_samples"], row["utterance_key"]))
    batches = []
    for samples in sorted({row["original_num_samples"] for row in ordered}):
        indices = [index for index, row in enumerate(ordered) if row["original_num_samples"] == samples]
        batches.extend(indices[start : start + args.batch_size] for start in range(0, len(indices), args.batch_size))
    loader = torch.utils.data.DataLoader(
        InputDataset(ordered),
        batch_sampler=batches,
        num_workers=args.workers,
        collate_fn=identity_collate,
        persistent_workers=args.workers > 0,
    )
    for batch in tqdm(loader, total=len(batches), desc="GRID VAE / CAM++ / REPA", unit="batch"):
        rows, waves, speaker_features, videos = zip(*batch)
        waveform = torch.from_numpy(np.stack(waves)).to(device)
        frames = rows[0]["latent_frames"]
        padded = F.pad(waveform[:, None], (0, frames * 400 - waveform.shape[-1]))
        hidden = vae.pre_block(vae.encoder(padded).transpose(1, 2))
        mu, logvar = vae.fc_mu(hidden), vae.fc_var(hidden).clamp(-12, 12)
        noise = torch.stack(
            [
                torch.randn(
                    mu[index].shape,
                    device=device,
                    dtype=mu.dtype,
                    generator=torch.Generator(device=device).manual_seed(row["posterior_seed"]),
                )
                for index, row in enumerate(rows)
            ]
        )
        latents = (mu + torch.exp(0.5 * logvar) * noise).float().cpu().numpy()
        chunks = torch.cat(speaker_features).to(device=device, dtype=torch.float32)
        embeddings = campplus(chunks).float()
        # GRID clips are below 10 seconds, so exactly one circularly padded
        # CAM++ chunk corresponds to each original unmasked waveform.
        if embeddings.shape[0] != len(rows):
            raise RuntimeError("Unexpected multiple CAM++ chunks for a GRID clip")
        embeddings = F.normalize(embeddings, dim=-1).cpu().numpy().astype(np.float32)
        inputs = extractor(
            list(waves), sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt"
        )
        teacher_hidden = teacher(
            input_values=inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)
        ).last_hidden_state
        teacher_lens = teacher._get_feat_extract_output_lengths(inputs.attention_mask.sum(dim=1))
        for index, row in enumerate(rows):
            video = (
                F.interpolate(
                    torch.from_numpy(videos[index]).T[None],
                    size=row["latent_frames"],
                    mode="linear",
                    align_corners=False,
                )[0]
                .T.contiguous()
                .numpy()
            )
            feature = teacher_hidden[index, : int(teacher_lens[index])].float().cpu().numpy().astype(np.float16)
            for key, array in (
                ("latent", latents[index]),
                ("video_40hz", video),
                ("speaker", embeddings[index]),
                ("repa", feature),
            ):
                atomic_save_npy(root / row[f"{key}_relative_path"], array)
            atomic_save_json(
                state_path(root, row),
                {
                    "spec_sha256": spec_sha,
                    "utterance_key": row["utterance_key"],
                    "source_audio_sha256": sha256_file(row["audio_path"]),
                    "source_video_sha256": sha256_file(row["video_source_path"]),
                    "features": verify_features(root, row),
                },
            )


def audit(args):
    root = args.output
    spec = read_json(root / "spec.json")
    spec_sha = sha256_file(root / "spec.json")
    rows = read_records(root)
    expected_files = {key: set() for key in FEATURE_KEYS}
    for row in tqdm(rows, desc="Full GRID feature audit", unit="clip"):
        if not verified_state(root, row, spec_sha):
            raise RuntimeError(f"Incomplete feature extraction: {row['utterance_key']}")
        state = read_json(state_path(root, row))
        if (
            sha256_file(row["audio_path"]) != state["source_audio_sha256"]
            or sha256_file(row["video_source_path"]) != state["source_video_sha256"]
        ):
            raise RuntimeError(f"Source feature checksums changed: {row['utterance_key']}")
        for key in FEATURE_KEYS:
            expected_files[key].add(root / row[f"{key}_relative_path"])
    for key, expected in expected_files.items():
        directory = "latents" if key == "latent" else "speaker_embeddings" if key == "speaker" else key
        if set((root / directory).rglob("*.npy")) != expected:
            raise RuntimeError(f"Unexpected extra/missing files in {root / directory}")
    for name, entry in spec["manifests"].items():
        if sha256_file(root / "manifests" / name) != entry["sha256"]:
            raise RuntimeError(f"Changed manifest: {name}")
    for key in ("normalization", "vocab"):
        if sha256_file(spec[key]["path"]) != spec[key]["sha256"]:
            raise RuntimeError(f"Changed pretrained {key}")
    contract = {**spec, "complete": True}
    immutable_json(root / "data_contract.json", contract)
    complete = {
        "schema_version": 1,
        "complete": True,
        "contract_sha256": sha256_file(root / "data_contract.json"),
        "split_counts": spec["split_counts"],
        "verified_count": len(rows),
        "train_val_overlap": 0,
        "latent_frame_counts": dict(Counter(row["latent_frames"] for row in rows)),
        "ctc_infeasible_count": sum(not row["ctc_feasible_40hz"] for row in rows),
        "source_and_feature_sha256_verified": True,
    }
    immutable_json(root / "complete.json", complete)
    print("GRID cache complete:", json.dumps(complete), flush=True)


def main():
    prefix = os.environ.get("ROOT_PREFIX", "")
    home = Path(f"{prefix}/zjw524")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("manifest", "extract", "audit", "all"), default="all")
    parser.add_argument(
        "--source-root", type=Path, default=home / "projects/aligndit_project_gird/alignDiT_baseline/AlignDiT/data_grid"
    )
    parser.add_argument("--output", type=Path, default=home / "projects/data/GRID_mmdit_svae")
    parser.add_argument(
        "--normalization",
        type=Path,
        default=home / "projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json",
    )
    parser.add_argument("--vocab", type=Path, default=home / "projects/data/CelebVDub_char/vocab.txt")
    parser.add_argument(
        "--vae-spec",
        type=Path,
        default=home / "projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/spec.json",
    )
    parser.add_argument("--vae-repo", type=Path, default=home / "projects/alignDiT_idea6/papers_codes/Semantic-VAE")
    parser.add_argument(
        "--vae-checkpoint",
        type=Path,
        default=home / "projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k",
    )
    parser.add_argument(
        "--speaker-checkpoint",
        type=Path,
        default=home
        / "projects/alignDiT_idea6/my_papers_code/AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding_adaptive_band_repa_wavlm_base_plus_chem/data_chem/pretrained_models/campplus/campplus_cn_en_common.pt",
    )
    parser.add_argument("--wavlm-dir", type=Path, default=home / "projects/data/wavlm-base-plus")
    parser.add_argument("--limit-per-split", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.limit_per_split < 0 or args.batch_size < 1 or args.workers < 0:
        parser.error("limit/workers must be nonnegative and batch-size must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    with (args.output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.stage in ("manifest", "all"):
            manifest(args)
        if args.stage in ("extract", "all"):
            extract(args)
        if args.stage in ("audit", "all"):
            audit(args)


if __name__ == "__main__":
    main()
