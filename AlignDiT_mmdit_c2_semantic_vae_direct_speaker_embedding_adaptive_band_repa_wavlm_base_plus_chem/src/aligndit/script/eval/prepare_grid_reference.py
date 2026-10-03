"""Freeze one GRID reference per CelebV-Dub target and extract its prompt caches.

``--plan-only`` performs all selection and source checks on CPU. The immutable
plan is reused by the normal invocation; only a complete, verified extraction
publishes ``pairs.jsonl`` for inference. All runtime artifacts live outside the
source repository. Existing artifacts must match their recorded hashes.
"""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm

from aligndit.model.speaker_embedding import validate_speaker_embedding_array
from aligndit.script.eval.semantic_vae_decoder import (
    HOP_LENGTH,
    LATENT_DIM,
    SAMPLE_RATE,
    load_semantic_vae,
    read_json_object,
    sha256_file,
)


EXPECTED_COUNT = 213
EXPECTED_SPEAKERS = tuple(f"s{i}" for i in range(1, 35) if i != 21)
PROTOCOL = "celebvdub213_grid_one_reference_v1"
BASE_POSTERIOR_SEED = 666


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def jsonl_bytes(rows: list[dict]) -> bytes:
    return ("".join(canonical_json(row) + "\n" for row in rows)).encode("utf-8")


def stable_reference_seed(ref_id: str, base_seed: int = BASE_POSTERIOR_SEED) -> int:
    """Same SHA256-to-int formula as training, with an external reference ID."""
    parts = ref_id.split("/")
    if len(parts) != 3 or parts[0] != "grid" or any(p in {"", ".", ".."} for p in parts) or "\\" in ref_id:
        raise ValueError(f"Invalid GRID reference ID: {ref_id!r}")
    if not 0 <= base_seed < 2**63:
        raise ValueError("posterior base seed must lie in [0, 2**63)")
    digest = hashlib.sha256(f"{base_seed}:{ref_id}".encode()).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)


def atomic_bytes(path: Path, payload: bytes, *, replace: bool = False) -> None:
    """Publish atomically, refusing to change any immutable existing artifact."""
    if path.exists() and not replace:
        if path.read_bytes() != payload:
            raise FileExistsError(f"Existing artifact differs; select a new output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: dict, *, replace: bool = False) -> None:
    atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"), replace=replace)


def read_grid_transcript(path: Path) -> str:
    # These .lab files contain complete text, not timestamps. In particular,
    # do not apply alignments from a different (untrimmed) GRID waveform.
    tokens = path.read_text(encoding="utf-8").strip().lower().split()
    # A minority of the local labels include MFA-style silence markers.
    # They are annotations, not words spoken in GRID's six-word sentences.
    text = " ".join(token for token in tokens if token not in {"sp", "sil", "<sil>"})
    if len(text.split()) != 6 or not all(word.isalpha() for word in text.split()):
        raise ValueError(f"Expected a six-word GRID transcript: {path}: {text!r}")
    return text


def grid_inventory(audio_root: Path, transcript_root: Path) -> dict[str, list[dict]]:
    inventory = {}
    for speaker in sorted(EXPECTED_SPEAKERS):
        directory = audio_root / speaker
        if not directory.is_dir() or not (transcript_root / speaker).is_dir():
            raise FileNotFoundError(f"Missing GRID audio/transcript directory for {speaker}")
        items = []
        for audio in sorted(directory.glob("*.wav")):
            lab = transcript_root / speaker / f"{speaker}-{audio.stem}.lab"
            if lab.is_file():
                items.append({"speaker": speaker, "audio": audio, "lab": lab, "clip": audio.stem})
        if not items:
            raise RuntimeError(f"No GRID references with transcripts for {speaker}")
        inventory[speaker] = items
    return inventory


def select_references(inventory: dict[str, list[dict]], count: int, seed: int) -> list[dict]:
    """Balanced speakers, independently shuffled utterances, no ref reuse."""
    if count <= 0 or not inventory:
        raise ValueError("Reference count and inventory must be nonempty")
    rng = random.Random(seed)
    speakers = sorted(inventory)
    rng.shuffle(speakers)
    assignments = [speakers[index % len(speakers)] for index in range(count)]
    rng.shuffle(assignments)
    needed = Counter(assignments)
    choices = {}
    for speaker in sorted(inventory):
        ordered = sorted(inventory[speaker], key=lambda item: item["clip"])
        if len(ordered) < needed[speaker]:
            raise ValueError(f"Insufficient distinct GRID utterances for {speaker}")
        choices[speaker] = iter(rng.sample(ordered, needed[speaker]))
    return [next(choices[speaker]) for speaker in assignments]


def build_plan(args: argparse.Namespace) -> tuple[list[dict], dict, dict]:
    # All imports stay within this snapshot; no other AlignDiT tree is added
    # to sys.path, even though the original cache engine lives elsewhere.
    from aligndit.script.eval.infer_celebvdub_semantic_vae_s1 import (
        load_composed_config,
        load_normalization,
        load_test_records,
    )
    from aligndit.script.misc.extract_campplus_speaker_embeddings import (
        EXPECTED_CHECKPOINT_SHA256,
        MODEL_ID,
    )

    config = load_composed_config(args.config.resolve(strict=True))
    cache_root = args.cache_root.resolve(strict=True)
    manifest = (args.manifest or cache_root / "manifests/test.jsonl").resolve(strict=True)
    test_list = args.test_list.resolve(strict=True)
    target_rows = load_test_records(cache_root, test_list, manifest)
    normalization = (args.normalization or Path(config.datasets.normalization_path)).resolve(strict=True)
    load_normalization(normalization)
    normalization_sha = sha256_file(normalization)
    if normalization_sha != config.datasets.expected_normalization_sha256:
        raise RuntimeError("Normalization hash differs from the training config")
    speaker_sha = sha256_file(args.speaker_checkpoint)
    if speaker_sha != EXPECTED_CHECKPOINT_SHA256 or speaker_sha != config.datasets.speaker_embedding_checkpoint_sha256:
        raise RuntimeError("CAM++ checkpoint hash differs from the training config")
    if config.datasets.speaker_embedding_model_id != MODEL_ID or int(config.model.arch.speaker_dim) != 192:
        raise RuntimeError("Expected the current 192-D CAM++ speaker-conditioned model")
    spec_path = cache_root / "state/latents/spec.json"
    spec = read_json_object(spec_path)
    if spec.get("extraction", {}).get("protocol") != "semantic_vae_posterior_sample_v1":
        raise RuntimeError("Unsupported Semantic-VAE latent-cache protocol")
    vae_checkpoint = args.semantic_vae_checkpoint.resolve(strict=True) / "dac/ema_state_dict.pth"
    vae_sha = sha256_file(vae_checkpoint)
    if vae_sha != spec["checkpoint"]["ema_sha256"]:
        raise RuntimeError("Semantic-VAE EMA differs from the training latent cache")

    grid_root = args.grid_root.resolve(strict=True)
    audio_root = (args.grid_audio_root or grid_root / "Grid_dataset_Raw/audio_25k").resolve(strict=True)
    transcript_root = (
        args.grid_transcript_root or grid_root / "Grid_resample_ABS/Grid_Wav_22050_Abs"
    ).resolve(strict=True)
    inventory = grid_inventory(audio_root, transcript_root)
    selected = select_references(inventory, EXPECTED_COUNT, args.seed)
    target_audio_root = args.target_audio_root.resolve(strict=True)
    output = args.output_dir.resolve()
    rows = []
    for target, ref in zip(target_rows, selected):
        target_id = target["utterance_key"].removeprefix("celebvdub/")
        if len(target_id.split("/")) != 3 or target_id.split("/")[0] != "test":
            raise ValueError(f"Unexpected target ID: {target_id}")
        frames, samples = int(target["latent_frames"]), int(target["original_num_samples"])
        if not (frames - 1) * HOP_LENGTH < samples <= frames * HOP_LENGTH:
            raise ValueError(f"Invalid target length: {target_id}")
        video_path = (cache_root / target["video_40hz_relative_path"]).resolve(strict=True)
        video = np.load(video_path, allow_pickle=False)
        if video.shape != (frames, 1024) or video.dtype != np.float32 or not np.isfinite(video).all():
            raise ValueError(f"Invalid 40-Hz target video: {video_path}")
        target_audio = (target_audio_root / target["audio_relative_path"]).resolve(strict=True)
        ref_audio = ref["audio"].resolve(strict=True)
        source_info = sf.info(ref_audio)
        if source_info.frames <= 0 or source_info.samplerate != 25000 or source_info.channels != 1:
            raise ValueError(f"Expected nonempty mono 25-kHz GRID source: {ref_audio}")
        ref_samples = math.ceil(source_info.frames * SAMPLE_RATE / source_info.samplerate)
        ref_id = f"grid/{ref['speaker']}/{ref['clip']}"
        ref_base = Path(ref["speaker"]) / ref["clip"]
        rows.append({
            "pair_id": f"{target_id}__{ref_id}",
            "target_id": target_id,
            "target_text": target["text"],
            "target_video_path": str(video_path),
            "target_video_sha256": sha256_file(video_path),
            "target_frames": frames,
            "target_num_samples": samples,
            "target_gt_audio": str(target_audio),
            "target_gt_audio_sha256": sha256_file(target_audio),
            "ref_id": ref_id,
            "ref_speaker_id": ref["speaker"],
            "ref_source_audio": str(ref_audio),
            "ref_source_audio_sha256": sha256_file(ref_audio),
            "ref_source_sample_rate": source_info.samplerate,
            "ref_source_num_samples": source_info.frames,
            "ref_transcript_path": str(ref["lab"].resolve(strict=True)),
            "ref_transcript_sha256": sha256_file(ref["lab"]),
            "ref_text": read_grid_transcript(ref["lab"]),
            "ref_audio": str(output / "audio" / ref_base.with_suffix(".wav")),
            "ref_num_samples": ref_samples,
            "ref_frames": math.ceil(ref_samples / HOP_LENGTH),
            "ref_latent_path": str(output / "latents" / ref_base.with_suffix(".npy")),
            "ref_speaker_path": str(output / "speakers" / ref_base.with_suffix(".npy")),
            "posterior_seed": stable_reference_seed(ref_id),
        })
    if len({row["target_id"] for row in rows}) != EXPECTED_COUNT or len({row["ref_id"] for row in rows}) != EXPECTED_COUNT:
        raise RuntimeError("Expected exactly 213 unique targets and 213 distinct reference clips")
    metadata = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "count": EXPECTED_COUNT,
        "pairing_seed": args.seed,
        "pairing_algorithm": "sorted_inventory_python_random_balanced_speakers_without_reference_replacement_v1",
        "speaker_counts": dict(sorted(Counter(row["ref_speaker_id"] for row in rows).items())),
        "excluded_speakers": {"s21": "no corresponding local reference transcripts"},
        "grid_audio_root": str(audio_root),
        "grid_transcript_root": str(transcript_root),
        "inventory_counts": {speaker: len(items) for speaker, items in inventory.items()},
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "test_list": str(test_list),
        "test_list_sha256": sha256_file(test_list),
        "cache_root": str(cache_root),
        "cache_spec_sha256": sha256_file(spec_path),
        "normalization_path": str(normalization),
        "normalization_sha256": normalization_sha,
        "semantic_vae_checkpoint": str(vae_checkpoint),
        "semantic_vae_checkpoint_sha256": vae_sha,
        "semantic_vae_source": spec["semantic_vae_source"],
        "speaker_checkpoint": str(args.speaker_checkpoint.resolve()),
        "speaker_checkpoint_sha256": speaker_sha,
        "speaker_model_id": MODEL_ID,
        "plan_sha256": hashlib.sha256(jsonl_bytes(rows)).hexdigest(),
        "target_duration_protocol": "existing_40hz_cache_GT_duration_metadata_known_target_duration",
        "cache_contract": {
            "reference_scope": "complete_source_utterance_no_cropping_no_alignment_time_axis",
            "transcript_processing": "lowercase_whitespace_collapse_remove_sp_sil_annotations_verify_six_words",
            "canonical_waveform": "mono_16000Hz_PCM16_from_raw_25000Hz_torchaudio_resample",
            "both_features_from": "exact_canonical_ref_audio_file",
            "latent_protocol": "semantic_vae_posterior_sample_v1",
            "sample_rate": SAMPLE_RATE,
            "hop_length": HOP_LENGTH,
            "latent_dtype": "float32",
            "latent_layout": "time,64",
            "latent_normalized": False,
            "base_posterior_seed": BASE_POSTERIOR_SEED,
            "posterior_seed_formula": "little_endian_SHA256_first8(666:ref_id)_mod_(2**63-1)",
            "posterior_noise": "torch.randn with per-reference CUDA Generator",
            "posterior_logvar_clamp": [-12, 12],
            "waveform_right_padding": "zero_to_next_400_sample_multiple",
            "speaker_dim": 192,
            "speaker_dtype": "float32",
            "speaker_chunk_seconds": 10,
            "speaker_max_seconds": 90,
            "speaker_padding": "repeat_complete_utterance_to_chunk_multiple",
            "speaker_aggregation": "arithmetic_mean_then_l2_normalize",
        },
    }
    return rows, metadata, spec


def record_path(output: Path, row: dict) -> Path:
    return output / "records" / Path(row["ref_id"]).relative_to("grid").with_suffix(".json")


def validate_cached_record(record: dict, planned: dict) -> None:
    for key, value in planned.items():
        if record.get(key) != value:
            raise RuntimeError(f"Cached reference plan mismatch for {planned['ref_id']}: {key}")
    for key in ("ref_audio", "ref_latent", "ref_speaker"):
        path_key = key if key == "ref_audio" else f"{key}_path"
        path = Path(record[path_key])
        if not path.is_file() or sha256_file(path) != record.get(f"{key}_sha256"):
            raise RuntimeError(f"Reference cache hash mismatch: {path}")
    audio, sample_rate = sf.read(record["ref_audio"], dtype="float32", always_2d=True)
    if sample_rate != SAMPLE_RATE or audio.shape != (record["ref_num_samples"], 1) or not np.isfinite(audio).all():
        raise ValueError(f"Invalid canonical reference waveform: {record['ref_audio']}")
    latent = np.load(record["ref_latent_path"], allow_pickle=False)
    if latent.dtype != np.float32 or latent.shape != (record["ref_frames"], LATENT_DIM) or not np.isfinite(latent).all():
        raise ValueError(f"Invalid reference latent: {record['ref_latent_path']}")
    validate_speaker_embedding_array(np.load(record["ref_speaker_path"], allow_pickle=False), source=record["ref_id"])


def write_canonical_waveform(row: dict) -> None:
    path = Path(row["ref_audio"])
    if path.exists():
        raise FileExistsError(f"Untracked waveform exists; refusing to replace: {path}")
    audio, source_rate = sf.read(row["ref_source_audio"], dtype="float32", always_2d=True)
    audio = torch.from_numpy(audio.mean(axis=1))
    if not torch.isfinite(audio).all():
        raise ValueError(f"Nonfinite GRID source: {row['ref_source_audio']}")
    audio = torchaudio.functional.resample(audio, source_rate, SAMPLE_RATE)
    if audio.numel() != row["ref_num_samples"]:
        raise ValueError(f"Unexpected resampled reference duration: {row['ref_id']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.wav")
    try:
        sf.write(temporary, audio.numpy(), SAMPLE_RATE, subtype="PCM_16")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def posterior_stats(model, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = model.pre_block(model.encoder(waveform).transpose(1, 2))
    return model.fc_mu(hidden), model.fc_var(hidden).clamp(-12, 12)


@torch.inference_mode()
def extract_reference(row: dict, vae, campplus, device: torch.device) -> dict:
    from aligndit.script.misc.extract_campplus_speaker_embeddings import (
        AudioFeatureDataset,
        ExtractionItem,
        atomic_save_npy,
    )

    write_canonical_waveform(row)
    waveform, sample_rate = torchaudio.load(row["ref_audio"])
    if sample_rate != SAMPLE_RATE or waveform.shape != (1, row["ref_num_samples"]):
        raise ValueError("Canonical WAV decoding mismatch")
    waveform = F.pad(waveform.unsqueeze(0), (0, row["ref_frames"] * HOP_LENGTH - waveform.shape[-1]))
    mu, logvar = posterior_stats(vae, waveform.to(device=device, dtype=torch.float32))
    generator = torch.Generator(device=device).manual_seed(row["posterior_seed"])
    noise = torch.randn(mu.shape, dtype=mu.dtype, device=device, generator=generator)
    latent = (mu + torch.exp(0.5 * logvar) * noise).squeeze(0).cpu().numpy().astype(np.float32, copy=False)
    if latent.shape != (row["ref_frames"], LATENT_DIM) or not np.isfinite(latent).all():
        raise RuntimeError(f"Invalid extracted latent: {row['ref_id']}")
    item = ExtractionItem(row["ref_audio"], row["ref_speaker_path"], "grid_reference")
    _, features, error = AudioFeatureDataset([item])[0]
    if error:
        raise RuntimeError(f"CAM++ preprocessing failed: {row['ref_id']}: {error}")
    embedding = campplus(features.to(device=device, dtype=torch.float32)).float().cpu().mean(dim=0)
    embedding = F.normalize(embedding, dim=0).numpy().astype(np.float32, copy=False)
    validate_speaker_embedding_array(embedding, source=row["ref_id"])
    atomic_save_npy(Path(row["ref_latent_path"]), latent)
    atomic_save_npy(Path(row["ref_speaker_path"]), embedding)
    record = dict(row)
    for key in ("ref_audio", "ref_latent", "ref_speaker"):
        path_key = key if key == "ref_audio" else f"{key}_path"
        record[f"{key}_sha256"] = sha256_file(record[path_key])
    validate_cached_record(record, row)
    return record


def prepare(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    source_repository = Path(__file__).resolve().parents[5]
    if output.is_relative_to(source_repository):
        raise ValueError("Reference runtime outputs must be outside my_papers_code")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, metadata, spec = build_plan(args)
        metadata_path = output / "metadata.json"
        previous = read_json_object(metadata_path) if metadata_path.exists() else None
        if previous is not None:
            for key, value in metadata.items():
                if previous.get(key) != value:
                    raise RuntimeError(f"Existing GRID preparation contract differs: {key}; choose a new output directory")
        atomic_bytes(output / "plan.jsonl", jsonl_bytes(rows))
        if previous is None:
            write_json(metadata_path, {**metadata, "status": "planned"})
        print(f"Frozen GRID plan: {len(rows)} targets, {len(metadata['speaker_counts'])} speakers, {output}", flush=True)
        if args.plan_only:
            return
        completed, pending = {}, []
        for row in rows:
            path = record_path(output, row)
            if path.exists():
                record = read_json_object(path)
                validate_cached_record(record, row)
                completed[row["ref_id"]] = record
            else:
                for key in ("ref_audio", "ref_latent_path", "ref_speaker_path"):
                    if Path(row[key]).exists():
                        raise FileExistsError(f"Artifact has no verified record; refusing to replace: {row[key]}")
                pending.append(row)
        if pending:
            device = torch.device(args.device)
            if device.type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("Formal posterior-sample extraction requires CUDA; use --plan-only for CPU validation")
            torch.cuda.set_device(device)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = False
            torch.use_deterministic_algorithms(False)
            from aligndit.script.misc.extract_campplus_speaker_embeddings import load_campplus

            vae, vae_metadata = load_semantic_vae(
                repo=args.semantic_vae_repo, checkpoint_root=args.semantic_vae_checkpoint,
                cache_spec=spec, device=torch.device("cpu"),
            )
            del vae.decoder  # Extraction needs only the already verified posterior modules.
            vae.to(device=device, dtype=torch.float32)
            campplus = load_campplus(args.speaker_checkpoint, device)
            with torch.inference_mode():
                posterior_stats(vae, torch.zeros(1, 1, HOP_LENGTH, device=device))
            write_json(metadata_path, {**metadata, "status": "extracting"}, replace=True)
            for row in tqdm(pending, desc="GRID reference VAE + CAM++", unit="utterance"):
                record = extract_reference(row, vae, campplus, device)
                write_json(record_path(output, row), record)
                completed[row["ref_id"]] = record
            metadata["extraction_runtime"] = {
                "torch": torch.__version__, "torchaudio": torchaudio.__version__,
                "numpy": np.__version__, "device": str(device),
                "device_name": torch.cuda.get_device_name(device),
                "cuda": torch.version.cuda, "vae_loader": vae_metadata,
            }
        elif previous and "extraction_runtime" in previous:
            metadata["extraction_runtime"] = previous["extraction_runtime"]
        ordered = [completed[row["ref_id"]] for row in rows]
        for record, planned in zip(ordered, rows):
            validate_cached_record(record, planned)
        manifest_path = output / "pairs.jsonl"
        atomic_bytes(manifest_path, jsonl_bytes(ordered))
        write_json(metadata_path, {
            **metadata, "status": "complete", "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path), "verified_count": len(ordered),
        }, replace=True)
        print(f"GRID reference preparation complete: {len(ordered)} verified pairs: {manifest_path}", flush=True)


def parse_args() -> argparse.Namespace:
    prefix = os.environ.get("ROOT_PREFIX", "")
    project = Path(__file__).resolve().parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=project / "src/aligndit/config/finetune_celebvdub_mm_c2_svae_speaker_adaptive_band_repa_wavlm_base_plus.yaml")
    parser.add_argument("--grid-root", type=Path, default=Path(f"{prefix}/zjw524/datasets/Grid_Dataset"))
    parser.add_argument("--grid-audio-root", type=Path)
    parser.add_argument("--grid-transcript-root", type=Path)
    parser.add_argument("--cache-root", type=Path, default=Path(f"{prefix}/zjw524/projects/data/CelebVDub_svae1000k_sample_seed666_fp32"))
    parser.add_argument("--test-list", type=Path, default=Path(f"{prefix}/zjw524/projects/data/celebvdub_test_s1.lst"))
    parser.add_argument("--target-audio-root", type=Path, default=Path(f"{prefix}/zjw524/projects/data/CelebVDub/audio"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--normalization", type=Path)
    parser.add_argument("--semantic-vae-repo", type=Path, default=Path(f"{prefix}/zjw524/projects/alignDiT_idea6/papers_codes/Semantic-VAE"))
    parser.add_argument("--semantic-vae-checkpoint", type=Path, default=Path(f"{prefix}/zjw524/projects/alignDiT_idea6/Semantic-VAE/semantic_vae_1000k"))
    parser.add_argument("--speaker-checkpoint", type=Path, default=Path(f"{prefix}/zjw524/projects/data/pretrained_models/3D-Speaker/speech_campplus_sv_zh_en_16k-common_advanced/campplus_cn_en_common.pt"))
    parser.add_argument("--seed", "--pairing-seed", dest="seed", type=int, default=0, help="Fixed reference pairing seed, independent of posterior seed 666")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--plan-only", action="store_true", help="Freeze and validate source selection without loading GPU models")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
