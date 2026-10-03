"""Build Chem's immutable fixed-posterior SVAE and time-aligned video caches.

The existing audited Chem split and estimated audiovisual alignment are reused;
no new alignment claim is made. Latents use the original fixed LibriSpeech
normalization at training time, never statistics fitted to Chem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch import nn


LATENT_FEATURE = "semantic_vae_posterior_sample_v1"
VIDEO_FEATURE = "avhubert_video_25hz_to_40hz_linear_align_corners_false_v1"
EMA_SHA256 = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
NORM_SHA256 = "65b8ab93520b88dc12492fe6ffb471d510bb77502d59d17eaa81e78e3d02c3f6"
EXPECTED_COUNTS = {"train": 5821, "val": 311, "test": 196}


def prefixed(relative: str) -> Path:
    return Path(f"{os.environ.get('ROOT_PREFIX', '')}/zjw524/{relative}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, content: bytes, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Immutable cache contract changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_json(path: Path, value: dict, *, immutable: bool = False) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(), immutable=immutable)


def write_jsonl(path: Path, rows: list[dict]) -> dict:
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    atomic_bytes(path, payload.encode(), immutable=True)
    return {
        "count": len(rows),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "path": f"manifests/{path.name}",
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_seed(key: str) -> int:
    digest = hashlib.sha256(f"666:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**63 - 1)


def inspect_record(item: tuple[int, dict], baseline: Path, vocabulary: dict[str, int]) -> dict:
    index, source = item
    split, video_id, clip_id = source["split"], source["video_id"], source["id"]
    audio = Path(source["audio_path"]).resolve(strict=True)
    video = baseline / "Chem/avhubert_video_feat" / split / video_id / f"{clip_id}.npy"
    info = sf.info(audio)
    if info.channels != 1 or info.samplerate != 16000 or info.frames != source["audio_samples"]:
        raise ValueError(f"Source audio differs from audited baseline: {audio}")
    array = np.load(video, allow_pickle=False)
    if array.ndim != 2 or array.shape[1] != 1024 or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"Invalid audited video: {video}")
    if abs(array.shape[0] / 25 - info.frames / 16000) > 0.020001:
        raise ValueError(f"Audited video/audio duration changed: {clip_id}")
    text = source["text"]
    if set(text) - vocabulary.keys():
        raise ValueError(f"Unseen transcript characters for {clip_id}: {set(text) - vocabulary.keys()}")
    ids = [vocabulary[c] for c in text]
    repeats = sum(a == b for a, b in pairwise(ids))
    frames = (info.frames + 399) // 400
    key = f"chem/{split}/{video_id}/{clip_id}"
    relative = f"{split}/{video_id}/{clip_id}"
    return {
        "utterance_key": key,
        "split": split,
        "subset": split,
        "video_id": video_id,
        "source_index": index,
        "text": text,
        "audio_path": str(audio),
        "audio_relative_path": relative + ".wav",
        "audio_sha256": sha256(audio),
        "duration_seconds": info.frames / 16000,
        "sample_rate": 16000,
        "num_channels": 1,
        "source_sample_rate": 16000,
        "source_num_channels": 1,
        "source_num_samples": info.frames,
        "original_num_samples": info.frames,
        "padded_num_samples": frames * 400,
        "latent_dim": 64,
        "latent_frames": frames,
        "posterior_seed": stable_seed(key),
        "latent_relative_path": "latents/" + relative + ".npy",
        "video_40hz_relative_path": "video_40hz/" + relative + ".npy",
        "video_relative_path": relative + ".npy",
        "video_source_path": str(video),
        "video_source_sha256": sha256(video),
        "video_dim": 1024,
        "video_frames_25hz": array.shape[0],
        "ctc_target_length": len(ids),
        "ctc_adjacent_repeats": repeats,
        "ctc_min_input_frames": len(ids) + repeats,
        "ctc_feasible_40hz": frames >= len(ids) + repeats,
    }


def prepare_manifest(args: argparse.Namespace) -> None:
    baseline = args.baseline_root.resolve(strict=True)
    complete = json.loads((baseline / "complete.json").read_text())
    if not complete.get("training_ready") or complete.get("counts") != EXPECTED_COUNTS:
        raise ValueError("Baseline Chem cache is not the audited complete split")
    if complete.get("visual_input") != "video_only" or complete.get("feature_dim") != 1024:
        raise ValueError("Chem requires pure-visual 1024-D AV-HuBERT features")
    normalization = args.normalization.resolve(strict=True)
    if sha256(normalization) != NORM_SHA256:
        raise ValueError("Original LibriSpeech train normalization SHA256 mismatch")
    source_records = json.loads((baseline / "records.json").read_text())
    vocabulary_path = baseline / "Chem_char/vocab.txt"
    characters = vocabulary_path.read_text().splitlines()
    vocabulary = {c: i for i, c in enumerate(characters)}
    if vocabulary.get(" ") != 0 or len(vocabulary) != len(characters):
        raise ValueError("Invalid source character vocabulary")
    atomic_bytes(args.cache_root / "manifests/vocab.txt", vocabulary_path.read_bytes(), immutable=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(lambda item: inspect_record(item, baseline, vocabulary), enumerate(source_records)))
    if dict(Counter(r["split"] for r in rows)) != EXPECTED_COUNTS:
        raise ValueError("Chem split changed")
    if len({r["utterance_key"] for r in rows}) != len(rows):
        raise ValueError("Duplicate Chem utterance")
    groups = {split: [r for r in rows if r["split"] == split] for split in EXPECTED_COUNTS}
    manifests = {"inventory.jsonl": write_jsonl(args.cache_root / "manifests/inventory.jsonl", rows)}
    for split, records in groups.items():
        manifests[f"{split}.jsonl"] = write_jsonl(args.cache_root / f"manifests/{split}.jsonl", records)
    ctc_counts = {
        split: {
            "total": len(records),
            "valid": sum(r["ctc_feasible_40hz"] for r in records),
            "excluded": sum(not r["ctc_feasible_40hz"] for r in records),
        }
        for split, records in groups.items()
    }
    meta = {
        "cache_schema_version": 1,
        "dataset": "Chem",
        "base_posterior_seed": 666,
        "dataset_root": str(baseline / "Chem/audio"),
        "split_counts": EXPECTED_COUNTS,
        "manifests": manifests,
        "ctc40_preflight": ctc_counts,
        "latent_spec": {
            "dimension": 64,
            "dtype": "float32",
            "frame_rate_hz": 40.0,
            "hop_length_samples": 400,
            "mode": "fixed_posterior_sample",
            "sample_rate": 16000,
        },
        "video_feature_spec": {"dimension": 1024, "dtype": "float32", "frame_rate_hz": 25},
        "video_alignment": complete["alignment"],
        "baseline_root": str(baseline),
        "baseline_complete_sha256": sha256(baseline / "complete.json"),
        "baseline_records_sha256": sha256(baseline / "records.json"),
        "normalization": {"path": str(normalization), "sha256": NORM_SHA256, "fitted_on": "LibriSpeech train"},
        "vocab_sha256": sha256(vocabulary_path),
        "total_latent_frames": sum(r["latent_frames"] for r in rows),
    }
    write_json(args.cache_root / "manifests/inventory_meta.json", meta, immutable=True)
    print(json.dumps({"manifests": manifests, "ctc": ctc_counts, "vocab_sha256": meta["vocab_sha256"]}), flush=True)


def valid_array(path: Path, frames: int, dim: int) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if array.shape != (frames, dim) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"Invalid cached array {path}: {array.shape}, {array.dtype}")
    return array


def save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as file:
        np.save(file, array, allow_pickle=False)
    os.replace(temporary, path)


def cache_video(args: argparse.Namespace, rows: list[dict]) -> None:
    def one(row: dict) -> dict:
        path = args.cache_root / row["video_40hz_relative_path"]
        source = Path(row["video_source_path"])
        if sha256(source) != row["video_source_sha256"]:
            raise RuntimeError(f"Source visual feature changed: {source}")
        array = valid_array(source, row["video_frames_25hz"], 1024)
        tensor = torch.from_numpy(array).T.unsqueeze(0)
        expected = F.interpolate(tensor, size=row["latent_frames"], mode="linear", align_corners=False)
        expected = expected.squeeze(0).T.contiguous().numpy()
        if path.exists():
            actual = valid_array(path, row["latent_frames"], 1024)
            if not np.array_equal(actual, expected):
                raise RuntimeError(f"Cached interpolated video changed: {path}")
        else:
            save_array(path, expected)
        return {
            "utterance_key": row["utterance_key"],
            "relative_path": row["video_40hz_relative_path"],
            "sha256": sha256(path),
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        hashes = list(pool.map(one, rows))
    write_jsonl(args.cache_root / "state/video_40hz/files.jsonl", hashes)
    completion(args, rows, "video_40hz", VIDEO_FEATURE, "total_target_frames")


class SemanticVaePosterior(nn.Module):
    def __init__(self, encoder_class: type[nn.Module], attention_class: type[nn.Module]):
        super().__init__()
        self.encoder = encoder_class(d_model=64, strides=[4, 4, 5, 5], d_latent=1024)
        self.pre_block = attention_class(1024, 64, num_heads=8)
        self.fc_mu = nn.Linear(64, 64)
        self.fc_var = nn.Linear(64, 64)

    def stats(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.pre_block(self.encoder(waveform).transpose(1, 2))
        return self.fc_mu(hidden), self.fc_var(hidden).clamp(-12, 12)


def load_posterior(args: argparse.Namespace, device: torch.device) -> SemanticVaePosterior:
    checkpoint_path = args.checkpoint_root / "dac/ema_state_dict.pth"
    if sha256(checkpoint_path) != EMA_SHA256:
        raise ValueError("Semantic-VAE 1000k EMA checksum mismatch")
    semantic_repo = args.semantic_vae_repo.resolve(strict=True)
    sys.path.insert(0, str(semantic_repo))
    import dac
    from dac.model.attn_proj import AttnProjection
    from dac.model.dac import Encoder

    if Path(dac.__file__).resolve().parent.parent != semantic_repo:
        raise RuntimeError("Imported incorrect Semantic-VAE snapshot")
    model = SemanticVaePosterior(Encoder, AttnProjection)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    if int(state["step"]) != 1000014 or not bool(state["initted"]):
        raise ValueError("Unexpected SVAE EMA step")
    prefixes = ("encoder.", "pre_block.", "fc_mu.", "fc_var.")
    posterior = {
        k.removeprefix("ema_model."): v
        for k, v in state.items()
        if k.startswith("ema_model.") and k.removeprefix("ema_model.").startswith(prefixes)
    }
    if len(posterior) != 145:
        raise ValueError("SVAE posterior schema changed")
    model.load_state_dict(posterior, strict=True)
    model.eval().requires_grad_(False).to(device)
    with torch.inference_mode():
        model.stats(torch.zeros(1, 1, 400, device=device))
    return model


@torch.inference_mode()
def encode(model: SemanticVaePosterior, path: Path, samples: int, seed: int, device: torch.device) -> np.ndarray:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != 16000 or audio.shape != (samples, 1) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid source waveform: {path}")
    frames = (samples + 399) // 400
    waveform = torch.from_numpy(audio.T.copy()).unsqueeze(0).to(device)
    waveform = F.pad(waveform, (0, frames * 400 - samples))
    mu, log_var = model.stats(waveform)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
    array = (mu + torch.exp(0.5 * log_var) * noise).squeeze(0).contiguous().cpu().numpy()
    if array.shape != (frames, 64) or array.dtype != np.float32 or not np.isfinite(array).all():
        raise RuntimeError("Semantic-VAE returned invalid latent")
    return array


def golden_test(args: argparse.Namespace, model: SemanticVaePosterior, device: torch.device) -> None:
    path = prefixed("datasets/LibriSpeech/train-clean-100/LibriSpeech/train-clean-100/103/1240/103-1240-0015.flac")
    expected = "e3de5ff47682f97e063c6aaeaee9cec195ebdb34e1bce964c4a10d2912114f3f"
    array = encode(model, path, 60960, 3920034511769737100, device)
    actual = hashlib.sha256(array.tobytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Original LibriSpeech posterior golden mismatch: {actual} != {expected}")
    write_json(
        args.cache_root / "state/latents/golden_test.json",
        {"passed": True, "sha256": actual, "torch_version": torch.__version__, "device": str(device)},
    )


def cache_latents(args: argparse.Namespace, rows: list[dict]) -> None:
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Use CUDA to preserve the original CUDA posterior RNG protocol")
    torch.backends.cuda.matmul.allow_tf32 = False
    # Match the original LibriSpeech/CelebV-Dub posterior cache protocol.
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    torch.use_deterministic_algorithms(False)
    torch.set_float32_matmul_precision("highest")
    model = load_posterior(args, device)
    golden_test(args, model, device)
    spec = {
        "feature": LATENT_FEATURE,
        "checkpoint_sha256": EMA_SHA256,
        "base_seed": 666,
        "manifest_sha256": sha256(args.cache_root / "manifests/inventory.jsonl"),
        "torch_version": str(torch.__version__),
        "precision": "float32",
        "matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
    }
    write_json(args.cache_root / "state/latents/spec.json", spec, immutable=True)
    started = time.monotonic()
    hashes = []
    for index, row in enumerate(rows):
        audio = Path(row["audio_path"])
        if sha256(audio) != row["audio_sha256"]:
            raise RuntimeError(f"Source audio changed: {audio}")
        path = args.cache_root / row["latent_relative_path"]
        proof_path = args.cache_root / "state/latents/records" / (row["audio_relative_path"] + ".json")
        if path.exists() and proof_path.exists():
            valid_array(path, row["latent_frames"], 64)
            proof = json.loads(proof_path.read_text())
            if proof.get("sha256") != sha256(path) or proof.get("audio_sha256") != row["audio_sha256"]:
                raise RuntimeError(f"Latent cache fingerprint changed: {path}")
        else:
            array = encode(model, audio, row["original_num_samples"], row["posterior_seed"], device)
            save_array(path, array)
            proof = {
                "utterance_key": row["utterance_key"],
                "relative_path": row["latent_relative_path"],
                "audio_sha256": row["audio_sha256"],
                "sha256": sha256(path),
            }
            write_json(proof_path, proof)
        hashes.append(proof)
        if index % 100 == 0:
            print(f"latent {index + 1}/{len(rows)} elapsed={time.monotonic() - started:.1f}s", flush=True)
    if args.limit == 0:
        write_jsonl(args.cache_root / "state/latents/files.jsonl", hashes)
        completion(args, rows, "latents", LATENT_FEATURE, "total_latent_frames")
    else:
        write_jsonl(args.cache_root / f"state/latents/smoke_{args.limit}_files.jsonl", hashes)


def completion(args: argparse.Namespace, rows: list[dict], state: str, feature: str, frame_key: str) -> None:
    if len(rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError("Cannot mark a partial cache complete")
    write_json(
        args.cache_root / f"state/{state}/complete.json",
        {
            "cache_schema_version": 1,
            "feature": feature,
            "selection": {"mode": "full"},
            "count": len(rows),
            "manifest_sha256": sha256(args.cache_root / "manifests/inventory.jsonl"),
            frame_key: sum(row["latent_frames"] for row in rows),
            "files_sha256": sha256(args.cache_root / f"state/{state}/files.jsonl"),
        },
    )
    print(f"Complete {state}: {len(rows)} records", flush=True)


def audit_cache(args: argparse.Namespace, rows: list[dict]) -> None:
    """Check full coverage, on-disk fingerprints and training normalization."""
    if sha256(args.normalization) != NORM_SHA256:
        raise RuntimeError("LibriSpeech normalization changed")
    norm = json.loads(args.normalization.read_text())
    mean, std = np.asarray(norm["mean"], np.float32), np.asarray(norm["std"], np.float32)
    latent_meta = json.loads((args.cache_root / "state/latents/complete.json").read_text())
    video_meta = json.loads((args.cache_root / "state/video_40hz/complete.json").read_text())
    indices = {}
    for stage, meta in (("latents", latent_meta), ("video_40hz", video_meta)):
        file_path = args.cache_root / f"state/{stage}/files.jsonl"
        if sha256(file_path) != meta["files_sha256"] or meta["count"] != len(rows):
            raise RuntimeError(f"Invalid {stage} completion fingerprint or count")
        index = {r["utterance_key"]: r for r in read_jsonl(file_path)}
        if set(index) != {row["utterance_key"] for row in rows}:
            raise RuntimeError(f"Incomplete {stage} key coverage")
        indices[stage] = index

    def one(row: dict) -> tuple[float, float, int]:
        latent_path = args.cache_root / row["latent_relative_path"]
        video_path = args.cache_root / row["video_40hz_relative_path"]
        latent = valid_array(latent_path, row["latent_frames"], 64)
        valid_array(video_path, row["latent_frames"], 1024)
        for stage, path in (("latents", latent_path), ("video_40hz", video_path)):
            if sha256(path) != indices[stage][row["utterance_key"]]["sha256"]:
                raise RuntimeError(f"Cached feature fingerprint changed: {path}")
        normalized = (latent - mean) / std
        if not np.isfinite(normalized).all():
            raise ValueError(f"Nonfinite normalized latent: {latent_path}")
        return float(normalized.min()), float(normalized.max()), latent.shape[0]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        values = list(pool.map(one, rows))
    report = {
        "complete": True,
        "counts": dict(Counter(r["split"] for r in rows)),
        "records": len(rows),
        "frames": sum(value[2] for value in values),
        "normalization_sha256": NORM_SHA256,
        "normalized_value_min": min(value[0] for value in values),
        "normalized_value_max": max(value[1] for value in values),
        "inventory_sha256": sha256(args.cache_root / "manifests/inventory.jsonl"),
    }
    write_json(args.cache_root / "state/audit.json", report)
    print(json.dumps(report), flush=True)
    # Bind the existing strict decoder loader to the same official codec.
    repo = args.semantic_vae_repo.resolve(strict=True)
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != "5bcca91fe8b65c0e52c5ee141968f98662dc4792":
        raise RuntimeError("Semantic-VAE source commit changed")
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Semantic-VAE source is dirty")
    metainfo = args.checkpoint_root / "metainfo.json"
    bigvgan = repo / json.loads(metainfo.read_text())["DAC"]["bigvgan_conf"]
    decoder_spec = {
        "checkpoint": {
            "ema_path": str(args.checkpoint_root / "dac/ema_state_dict.pth"),
            "ema_sha256": EMA_SHA256,
            "ema_step": 1000014,
            "metainfo_sha256": sha256(metainfo),
        },
        "semantic_vae_source": {"repo": str(repo), "commit": commit, "bigvgan_config_sha256": sha256(bigvgan)},
    }
    write_json(args.cache_root / "state/latents/decoder_spec.json", decoder_spec, immutable=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=prefixed(
            "projects/aligndit_project_gird/aligndit_project_chem/alignDiT_baseline/AlignDiT/data_chem_v2"
        ),
    )
    parser.add_argument("--cache-root", type=Path, default=Path("data_chem/svae1000k_sample_seed666_fp32"))
    parser.add_argument(
        "--normalization",
        type=Path,
        default=prefixed(
            "projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json"
        ),
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=prefixed("projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k"),
    )
    parser.add_argument(
        "--semantic-vae-repo", type=Path, default=prefixed("projects/alignDiT_idea6/papers_codes/Semantic-VAE")
    )
    parser.add_argument("--stages", default="manifest,video", help="Comma-separated manifest,video,latent,audit")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="Latent development limit; never marks complete")
    args = parser.parse_args()
    args.cache_root = args.cache_root.absolute()
    torch.set_num_threads(1)
    for stage in args.stages.split(","):
        if stage == "manifest":
            prepare_manifest(args)
            continue
        rows = read_jsonl(args.cache_root / "manifests/inventory.jsonl")
        if stage == "video":
            cache_video(args, rows)
        elif stage == "latent":
            cache_latents(args, rows[: args.limit] if args.limit else rows)
        elif stage == "audit":
            audit_cache(args, rows)
        else:
            raise ValueError(f"Unknown stage: {stage}")


if __name__ == "__main__":
    main()
