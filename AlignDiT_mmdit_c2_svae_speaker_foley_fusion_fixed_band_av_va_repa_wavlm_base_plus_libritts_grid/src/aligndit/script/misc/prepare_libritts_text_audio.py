"""Inventory, resumable extraction, and full audit for LibriTTS text/audio pretraining.

Run each stage explicitly. Extraction shards never publish a completion marker;
only a subsequent audit of every manifest record can make the cache trainable.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
import unicodedata
import wave
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch import nn

from aligndit.model.libritts_text_audio_dataset import (
    CACHE_SCHEMA, HOP_LENGTH, LATENT_DIM, NORMALIZATION_SHA256, SAMPLE_RATE,
    ctc_min_frames, load_cached_record, load_manifest, read_json, read_vocab,
    safe_path, sha256_file, validate_arrays,
)


SUBSETS = ("train-clean-100", "train-clean-360")
SEMANTIC_VAE_SHA256 = "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516"
CAMPPLUS_SHA256 = "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199"
TEXT_NORMALIZATION = "nfkc_smart_quotes_dashes_ellipsis_to_ascii_lower_whitespace_v1"
POSTERIOR_PROTOCOL = "semantic_vae_posterior_sample_v1"


def prefixed(relative):
    return Path(os.environ.get("ROOT_PREFIX", "") + "/zjw524") / relative


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def atomic_write(path, payload, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError(f"Refusing a symlink destination: {path}")
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Existing cache contract differs; use a new cache directory: {path}")
        return
    descriptor, name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def cache_lock(root, *, exclusive):
    root.mkdir(parents=True, exist_ok=True)
    # Linux NFS implements flock using byte-range locks: a shared lock requires
    # a readable descriptor, while an exclusive lock requires a writable one.
    with (root / ".cache.lock").open("a+") as stream:
        fcntl.flock(stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


@contextlib.contextmanager
def record_lock(root, key):
    # Bounded lock files also protect against accidentally overlapping shard layouts.
    lock_dir = root / ".record-locks"
    lock_dir.mkdir(exist_ok=True)
    bucket = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 1024
    with (lock_dir / str(bucket)).open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"', "‐": "-", "‑": "-", "–": "-", "—": "-", "…": "..."}))
    return " ".join(text.lower().split())


def posterior_seed(seed, key):
    # Same stable formula as the existing Semantic-VAE posterior cache protocol.
    if not 0 <= seed < 2**63:
        raise ValueError("Posterior seed must be in [0, 2**63)")
    return int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()[:8], "little") % (2**63 - 1)


def jsonl_bytes(records):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8") for row in records)


def build_inventory(args):
    root, source = args.cache_root.resolve(), args.dataset_root.resolve(strict=True)
    if args.limit_per_split is not None and "smoke" not in root.name.lower():
        raise ValueError("A limited inventory must use a separate cache directory containing 'smoke' in its name")
    with cache_lock(root, exclusive=True):
        if (root / "complete.json").exists():
            raise RuntimeError("Completed caches are immutable; use a new cache root")
        base_vocab = read_vocab(args.base_vocab)
        if sha256_file(args.normalization) != args.expected_normalization_sha256:
            raise RuntimeError("Normalization differs from the explicitly pinned S2c artifact")
        normalization = read_json(args.normalization)
        if normalization.get("channel_count") != LATENT_DIM or normalization.get("feature") != POSTERIOR_PROTOCOL:
            raise ValueError("Expected the fixed 64-dimensional Semantic-VAE posterior-sample normalization")
        records, failures, discovered = [], [], 0
        full_subset_counts = {}
        for subset in SUBSETS:
            directory = source / subset
            if not directory.is_dir():
                raise FileNotFoundError(f"Missing requested subset: {directory}")
            paths = sorted(directory.rglob("*.wav"))
            full_subset_counts[subset] = len(paths)
            print(f"inventory subset={subset} discovered_wavs={len(paths)}", flush=True)
            if args.limit_per_split is not None:
                paths = paths[: args.limit_per_split]
            else:
                expected_text = {path.with_suffix(".normalized.txt") for path in paths}
                extra_text = set(directory.rglob("*.normalized.txt")) - expected_text
                failures.extend({"path": str(path.relative_to(source)), "reason": "normalized text has no matching WAV"} for path in sorted(extra_text))
            for path in paths:
                discovered += 1
                relative = path.relative_to(source)
                try:
                    if len(relative.parts) != 4 or path.is_symlink():
                        raise ValueError("Expected a regular subset/speaker/chapter/utterance.wav file")
                    text_path = path.with_suffix(".normalized.txt")
                    raw_text_bytes = text_path.read_bytes()
                    raw_text = raw_text_bytes.decode("utf-8", errors="strict")
                    text = normalize_text(raw_text)
                    if not text:
                        raise ValueError("Empty normalized transcript")
                    with wave.open(str(path), "rb") as stream:
                        sr, samples = stream.getframerate(), stream.getnframes()
                        if sr != 24_000 or stream.getnchannels() != 1 or stream.getsampwidth() != 2 or samples <= 0:
                            raise ValueError("Expected nonempty 24 kHz mono PCM16 LibriTTS audio")
                    key = relative.with_suffix("").as_posix()
                    target_samples = math.ceil(samples * SAMPLE_RATE / sr)
                    records.append({
                        "utterance_key": key, "subset": subset, "speaker_id": relative.parts[1],
                        "audio_relative_path": relative.as_posix(), "text": text,
                        "normalized_text_sha256": hashlib.sha256(raw_text_bytes).hexdigest(),
                        "source_sample_rate": sr, "source_num_samples": samples,
                        "source_size_bytes": path.stat().st_size, "duration": samples / sr,
                        "resampled_num_samples": target_samples, "latent_frames": math.ceil(target_samples / HOP_LENGTH),
                        "posterior_seed": posterior_seed(args.seed, key),
                        "latent_relative_path": f"latents/{key}.npy", "speaker_relative_path": f"speakers/{key}.npy",
                        "receipt_relative_path": f"receipts/{key}.json",
                    })
                except Exception as error:
                    failures.append({"path": relative.as_posix(), "reason": f"{type(error).__name__}: {error}"})
                if discovered % 5000 == 0:
                    print(f"inventory checked={discovered} accepted={len(records)} failures={len(failures)}", flush=True)
        if args.limit_per_split is None and discovered != args.expected_count:
            failures.append({"path": ".", "reason": f"Expected {args.expected_count} WAV files, discovered {discovered}"})
        if args.limit_per_split is None and args.expected_count == 149736 and full_subset_counts != {"train-clean-100": 33236, "train-clean-360": 116500}:
            failures.append({"path": ".", "reason": f"Official subset counts do not match: {full_subset_counts}"})
        if failures:
            atomic_write(root / "inventory_failures.json", json_bytes({"discovered": discovered, "accepted": len(records), "failures": failures}))
            raise RuntimeError(f"Inventory rejected {len(failures)} source/count issues; inspect {root / 'inventory_failures.json'}")
        speakers = sorted({row["speaker_id"] for row in records})
        if not 0 < args.validation_speakers < len(speakers):
            raise ValueError(f"validation-speakers must be between 1 and {len(speakers) - 1}")
        validation = set(random.Random(args.seed).sample(speakers, args.validation_speakers))
        chars = list(base_vocab)
        chars.extend(sorted({char for row in records for char in row["text"]} - set(chars)))
        vocab = {char: index for index, char in enumerate(chars)}
        for row in records:
            row["split"] = "val" if row["speaker_id"] in validation else "train"
            row["ctc_min_frames"] = ctc_min_frames(row["text"], vocab)
            row["ctc_feasible"] = row["latent_frames"] >= row["ctc_min_frames"]
        payloads = {
            "inventory.jsonl": jsonl_bytes(records),
            "train.jsonl": jsonl_bytes([row for row in records if row["split"] == "train"]),
            "val.jsonl": jsonl_bytes([row for row in records if row["split"] == "val"]),
            "vocab.txt": ("\n".join(chars) + "\n").encode("utf-8"),
            "normalization.json": args.normalization.read_bytes(),
        }
        metadata = {
            "schema_version": CACHE_SCHEMA, "dataset_root": str(source), "seed": args.seed,
            "selection": {"mode": "full" if args.limit_per_split is None else "smoke", "limit_per_split": args.limit_per_split},
            "subsets": list(SUBSETS), "source_subset_counts": full_subset_counts,
            "counts": {"discovered": discovered, "accepted": len(records), "rejected": 0},
            "split_counts": dict(Counter(row["split"] for row in records)),
            "split_hours": {split: sum(row["duration"] for row in records if row["split"] == split) / 3600 for split in ("train", "val")},
            "validation_speakers": sorted(validation), "speaker_count": len(speakers),
            "ctc_infeasible": dict(Counter(row["split"] for row in records if not row["ctc_feasible"])),
            "max_duration": max(row["duration"] for row in records), "duration_filter": None,
            "text_normalization": TEXT_NORMALIZATION, "base_vocabulary": list(base_vocab),
            "base_vocab_sha256": sha256_file(args.base_vocab), "vocab_size": len(chars),
            "normalization_source": str(args.normalization.resolve()),
            "files": {name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()},
        }
        # Metadata is the final inventory commit marker. Partial writes are safe to retry.
        for name, payload in payloads.items():
            atomic_write(root / name, payload, immutable=True)
        atomic_write(root / "metadata.json", json_bytes(metadata), immutable=True)
        print(json.dumps({"cache_root": str(root), "counts": metadata["counts"], "split_counts": metadata["split_counts"], "hours": metadata["split_hours"], "vocab_size": len(chars)}, ensure_ascii=False), flush=True)


class SemanticPosterior(nn.Module):
    def __init__(self, encoder, attention):
        super().__init__()
        self.encoder = encoder(d_model=64, strides=[4, 4, 5, 5], d_latent=1024)
        self.pre_block = attention(1024, 64, num_heads=8)
        self.fc_mu, self.fc_var = nn.Linear(64, 64), nn.Linear(64, 64)

    def stats(self, waveform):
        hidden = self.pre_block(self.encoder(waveform).transpose(1, 2))
        return self.fc_mu(hidden), self.fc_var(hidden).clamp(-12, 12)


def load_codec(repo, checkpoint, device):
    repo = repo.resolve(strict=True)
    sys.path.insert(0, str(repo))
    try:
        import dac
        from dac.model.attn_proj import AttnProjection
        from dac.model.dac import Encoder
    finally:
        sys.path.pop(0)
    if Path(dac.__file__).resolve().parent.parent != repo:
        raise RuntimeError("Imported Semantic-VAE from the wrong repository")
    model = SemanticPosterior(Encoder, AttnProjection)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if not bool(state["initted"].item()) or int(state["step"].item()) != 1_000_014:
        raise ValueError("Unexpected Semantic-VAE EMA checkpoint step")
    posterior = {key.removeprefix("ema_model."): value for key, value in state.items() if key.startswith(tuple("ema_model." + name for name in ("encoder.", "pre_block.", "fc_mu.", "fc_var.")))}
    model.load_state_dict(posterior, strict=True)
    model.eval().requires_grad_(False).to(device=device, dtype=torch.float32)
    # Discard first-forward lazy kernel initialization, matching the existing cache.
    with torch.inference_mode():
        model.stats(torch.zeros(1, 1, HOP_LENGTH, device=device))
    return model


def make_extraction_spec(args, root):
    for path, expected in ((args.codec_checkpoint, SEMANTIC_VAE_SHA256), (args.speaker_checkpoint, CAMPPLUS_SHA256)):
        if sha256_file(path) != expected:
            raise RuntimeError(f"Incorrect pretrained weight SHA256: {path}")
    return {
        "schema_version": CACHE_SCHEMA, "metadata_sha256": sha256_file(root / "metadata.json"),
        "semantic_vae_checkpoint_sha256": SEMANTIC_VAE_SHA256,
        "speaker_checkpoint_sha256": CAMPPLUS_SHA256,
        "posterior_protocol": POSTERIOR_PROTOCOL, "dtype": "float32", "layout": "time_channel",
        "latent_dim": LATENT_DIM, "frame_rate": 40, "sample_rate": SAMPLE_RATE, "hop_length": HOP_LENGTH,
        "resample": "torchaudio.functional.resample_default_sinc_interp_hann_24000_to_16000",
        "padding": "right_zero_to_multiple_of_400_after_resampling",
        "speaker_protocol": "camplusplus_16k_80fbank_10sec_circular_chunks_full_utterance_mean_then_l2_v1",
        "speaker_dimension": 192, "speaker_max_seconds": None,
        "codec_sources": {name: sha256_file(args.codec_repo / name) for name in ("dac/model/dac.py", "dac/model/attn_proj.py")},
        "extractor_sha256": sha256_file(Path(__file__)),
        "torch_version": torch.__version__, "torchaudio_version": torchaudio.__version__,
        "matmul_tf32": False, "cudnn_tf32": True, "posterior_device": "cuda",
    }


def speaker_vector(model, waveform, device):
    from torchaudio.compliance.kaldi import fbank

    waveform = waveform.flatten()
    chunk_size = SAMPLE_RATE * 10
    total = math.ceil(waveform.numel() / chunk_size) * chunk_size
    chunks = waveform.repeat(math.ceil(total / waveform.numel()))[:total].reshape(-1, chunk_size)
    vectors = []
    for chunk in chunks:
        features = fbank(chunk.unsqueeze(0), num_mel_bins=80, sample_frequency=SAMPLE_RATE, dither=0.0)
        features = features - features.mean(dim=0, keepdim=True)
        vectors.append(model(features.unsqueeze(0).to(device)).squeeze(0))
    vector = F.normalize(torch.stack(vectors).mean(dim=0), dim=0)
    return vector.float().cpu().numpy()


def encode_record(args, metadata, row, codec, speaker, device):
    source = safe_path(Path(metadata["dataset_root"]), row["audio_relative_path"])
    if source.stat().st_size != row["source_size_bytes"]:
        raise RuntimeError(f"Source audio changed since inventory: {source}")
    text_path = source.with_suffix(".normalized.txt")
    if sha256_file(text_path) != row["normalized_text_sha256"]:
        raise RuntimeError(f"Source transcript changed since inventory: {text_path}")
    waveform, sr = torchaudio.load(source)
    if sr != row["source_sample_rate"] or waveform.shape != (1, row["source_num_samples"]):
        raise ValueError(f"Decoded audio differs from manifest: {source}")
    waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[1] != row["resampled_num_samples"]:
        raise RuntimeError(f"Unexpected resampling length for {source}")
    padded = F.pad(waveform, (0, row["latent_frames"] * HOP_LENGTH - waveform.shape[1]))
    with torch.inference_mode():
        mu, log_var = codec.stats(padded.unsqueeze(0).to(device=device, dtype=torch.float32))
        generator = torch.Generator(device=device).manual_seed(row["posterior_seed"])
        noise = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
        latent = (mu + torch.exp(0.5 * log_var) * noise).squeeze(0).float().cpu().numpy()
        embedding = speaker_vector(speaker, waveform, device)
    validate_arrays(latent, embedding, row["latent_frames"])
    return latent, embedding


def save_record(root, row, spec_sha, latent, speaker):
    receipt = {"utterance_key": row["utterance_key"], "extraction_spec_sha256": spec_sha}
    for kind, array in (("latent", latent), ("speaker", speaker)):
        buffer = io.BytesIO()
        np.save(buffer, array, allow_pickle=False)
        payload = buffer.getvalue()
        atomic_write(safe_path(root, row[f"{kind}_relative_path"]), payload)
        receipt[f"{kind}_sha256"] = hashlib.sha256(payload).hexdigest()
    # Receipt is published last, so interrupted extraction is recomputed on resume.
    atomic_write(safe_path(root, row["receipt_relative_path"]), json_bytes(receipt))


def extract_cache(args):
    root = args.cache_root.resolve(strict=True)
    if args.world_size <= 0 or not 0 <= args.rank < args.world_size:
        raise ValueError("Invalid shard rank/world size")
    if not torch.cuda.is_available():
        raise RuntimeError("Extraction uses CUDA to preserve the existing posterior sampling protocol")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Use a CUDA device for the fixed posterior cache")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    metadata, records, _ = load_manifest(root)
    spec = make_extraction_spec(args, root)
    # Separate short spec lock permits simultaneous shard initialization.
    with (root / ".spec.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        atomic_write(root / "extraction_spec.json", json_bytes(spec), immutable=True)
    spec_sha = sha256_file(root / "extraction_spec.json")
    with cache_lock(root, exclusive=False):
        if (root / "complete.json").exists():
            raise RuntimeError("Cache already complete; run audit to check it instead of overwriting it")
        # A shared cache lock excludes audit; rank locks reject duplicate shard writers.
        with (root / f".rank-{args.rank}.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assigned = records[args.rank::args.world_size]
            if args.limit is not None:
                assigned = assigned[:args.limit]
            codec = speaker = None
            restored = written = 0
            for row in assigned:
                with record_lock(root, row["utterance_key"]):
                    try:
                        load_cached_record(root, row, spec_sha)
                        restored += 1
                        continue
                    except Exception:
                        pass
                    if codec is None:
                        from aligndit.script.misc.extract_campplus_speaker_embeddings import load_campplus
                        codec = load_codec(args.codec_repo, args.codec_checkpoint, device)
                        speaker = load_campplus(args.speaker_checkpoint, device)
                    try:
                        latent, embedding = encode_record(args, metadata, row, codec, speaker, device)
                        save_record(root, row, spec_sha, latent, embedding)
                        written += 1
                    except Exception as error:
                        raise RuntimeError(f"Extraction failed at {row['utterance_key']}; completed files remain resumable; no completion was published") from error
                if (written + restored) % 100 == 0:
                    print(f"rank={args.rank} processed={written + restored}/{len(assigned)} written={written} reused={restored}", flush=True)
            print(json.dumps({"rank": args.rank, "assigned": len(assigned), "written": written, "reused": restored, "next": "Run audit after all shards finish"}), flush=True)


def audit_cache(args):
    root = args.cache_root.resolve(strict=True)
    with cache_lock(root, exclusive=True):
        # Re-audit invalidates an old marker until every record succeeds again.
        (root / "complete.json").unlink(missing_ok=True)
        metadata, records, _ = load_manifest(root)
        spec_sha = sha256_file(root / "extraction_spec.json")
        spec = read_json(root / "extraction_spec.json")
        if spec["metadata_sha256"] != sha256_file(root / "metadata.json"):
            raise RuntimeError("Extraction spec refers to a different inventory")
        failures, valid = [], 0
        expected = {kind: set() for kind in ("latents", "speakers", "receipts")}
        for row in records:
            for kind, field in (("latents", "latent_relative_path"), ("speakers", "speaker_relative_path"), ("receipts", "receipt_relative_path")):
                expected[kind].add(row[field])
            try:
                load_cached_record(root, row, spec_sha)
                valid += 1
            except Exception as error:
                failures.append({"utterance_key": row["utterance_key"], "reason": f"{type(error).__name__}: {error}"})
            if (valid + len(failures)) % 1000 == 0:
                print(f"audit checked={valid + len(failures)}/{len(records)} failures={len(failures)}", flush=True)
        extras = []
        for kind, wanted in expected.items():
            actual = {path.relative_to(root).as_posix() for path in (root / kind).rglob("*") if path.is_file() and not path.name.startswith(".tmp-")}
            extras.extend(sorted(actual - wanted))
        if extras:
            failures.append({"reason": "Unexpected files in feature cache", "paths": extras})
        report = {
            "complete": not failures, "expected": len(records), "valid": valid, "failures": failures,
            "metadata_sha256": sha256_file(root / "metadata.json"), "extraction_spec_sha256": spec_sha,
            "split_counts": metadata["split_counts"], "sha256_verified": True,
        }
        atomic_write(root / "audit_report.json", json_bytes(report))
        if failures:
            raise RuntimeError(f"Cache audit failed; {len(failures)} issues, see {root / 'audit_report.json'}")
        atomic_write(root / "complete.json", json_bytes(report))
        print(json.dumps({"complete": True, "expected": len(records), "valid": valid, "cache_root": str(root)}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inventory", "extract", "audit"):
        child = sub.add_parser(name)
        child.add_argument("--cache-root", type=Path, default=prefixed("projects/data/LibriTTS_text_audio_svae1000k_seed666_fp32"))
        if name == "inventory":
            child.add_argument("--dataset-root", type=Path, default=prefixed("datasets/LibriTTS_460h/LibriTTS"))
            child.add_argument("--base-vocab", type=Path, default=prefixed("projects/data/CelebVDub_char/vocab.txt"))
            child.add_argument("--normalization", type=Path, default=prefixed("projects/data/LibriSpeech_svae1000k_sample_seed666_fp32/state/latents/train_normalization.json"))
            child.add_argument("--expected-normalization-sha256", default=NORMALIZATION_SHA256)
            child.add_argument("--expected-count", type=int, default=149736)
            child.add_argument("--validation-speakers", type=int, default=20)
            child.add_argument("--seed", type=int, default=666)
            child.add_argument("--limit-per-split", type=int, help="Only for a separate smoke cache; never changes the full inventory")
        elif name == "extract":
            child.add_argument("--codec-repo", type=Path, default=prefixed("projects/alignDiT_idea6/papers_codes/Semantic-VAE"))
            child.add_argument("--codec-checkpoint", type=Path, default=prefixed("projects/alignDiT_idea6/Semantic-VAE/Semantic-VAE/semantic_vae_1000k/dac/ema_state_dict.pth"))
            child.add_argument("--speaker-checkpoint", type=Path, required=True)
            child.add_argument("--rank", type=int, default=int(os.environ.get("RANK", "0")))
            child.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
            child.add_argument("--device", default="cuda:" + os.environ.get("LOCAL_RANK", "0"))
            child.add_argument("--limit", type=int, help="Smoke only: cap this shard; audit still requires all manifest records")
    return parser.parse_args()


def main():
    args = parse_args()
    for name in ("limit", "limit_per_split"):
        if getattr(args, name, None) is not None and getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    {"inventory": build_inventory, "extract": extract_cache, "audit": audit_cache}[args.command](args)


if __name__ == "__main__":
    main()
