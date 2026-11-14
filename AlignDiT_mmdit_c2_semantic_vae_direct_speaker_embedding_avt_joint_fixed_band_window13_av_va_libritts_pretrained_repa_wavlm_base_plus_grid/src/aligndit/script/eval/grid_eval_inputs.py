"""GRID evaluation inputs copied from the audited reference protocol.

Only audio, latent, video and speaker cache readers are included. The target
snapshot retains its own fixed-band model and pinned Semantic-VAE decoder.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from aligndit.script.eval.semantic_vae_decoder import (
    LATENT_DIM, SAMPLE_RATE, read_json_object, sha256_file,
)

STAGES = ("audio", "latent", "video", "speaker")


def artifact_path(output: Path, row: dict, stage: str) -> Path:
    if stage not in STAGES:
        raise ValueError(f"Unsupported GRID evaluation stage: {stage}")
    relative = Path(row["audio_relative_path"])
    if stage == "audio":
        return output / "audio" / relative
    if stage == "latent":
        return output / row["latent_relative_path"]
    if stage == "video":
        return output / row["video_40hz_relative_path"]
    return output / "speakers" / relative.with_suffix(".npy")

def receipt_path(output: Path, row: dict, stage: str) -> Path:
    return output / "receipts" / stage / Path(row["audio_relative_path"]).with_suffix(".json")

def validate_artifact(path: Path, row: dict, stage: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f'Missing or symlinked {stage} artifact: {path}')
    if stage == 'audio':
        value, rate = sf.read(path, dtype='float32', always_2d=True)
        info = sf.info(path)
        if rate != SAMPLE_RATE or value.shape != (row['original_num_samples'], 1) or info.subtype != 'PCM_16':
            raise ValueError(f'Invalid canonical GRID audio: {path}')
    else:
        value = np.load(path, allow_pickle=False)
        if stage == 'speaker':
            from aligndit.model.speaker_embedding import validate_speaker_embedding_array
            validate_speaker_embedding_array(value, source=path)
        else:
            dimension = LATENT_DIM if stage == 'latent' else 1024
            if value.shape != (row['latent_frames'], dimension) or value.dtype != np.float32:
                raise ValueError(f'Invalid {stage} shape/dtype: {path}: {value.shape}, {value.dtype}')
    if not np.isfinite(value).all():
        raise ValueError(f'Nonfinite {stage} artifact: {path}')

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
    if stage in {"latent", "speaker"}:
        audio_receipt = read_json_object(receipt_path(output, row, "audio"))
        if receipt.get("audio_sha256") != audio_receipt.get("artifact_sha256"):
            raise RuntimeError(f"GRID feature is bound to different canonical audio: {artifact}")
    return True

def reference_target_text(reference: str, target: str) -> str:
    """Use the historical English prompt separator with independent transcripts."""
    if not reference.strip() or not target.strip():
        raise ValueError("Both reference and target transcripts must be nonempty")
    if len(reference[-1].encode("utf-8")) == 1:
        reference += " "
    return reference + " " + target

def make_sampling_inputs(reference: np.ndarray, video: np.ndarray, text: str, mean, std):
    """Align both modalities on one 40-Hz grid, including any sampler text tail."""
    if reference.ndim != 2 or reference.shape[1] != LATENT_DIM or len(reference) == 0:
        raise ValueError("Expected nonempty GRID latent [P,64]")
    if video.ndim != 2 or video.shape[1] != 1024 or len(video) == 0:
        raise ValueError("Expected nonempty target video [T,1024]")
    if not np.isfinite(reference).all() or not np.isfinite(video).all():
        raise ValueError("Non-finite reference/video feature")
    prompt_frames, target_frames = len(reference), len(video)
    # The char tokenizer emits one token per character, including unknown chars.
    # Match CFM_VT.sample's max(text length, prompt length)+1 lower bound exactly.
    requested_duration = prompt_frames + target_frames
    effective_duration = max(requested_duration, len(text) + 1, prompt_frames + 1)
    if effective_duration > 4096:
        raise ValueError("The combined prompt/target exceeds the sampler's 4096-frame limit")
    combined_video = np.zeros((effective_duration, 1024), dtype=np.float32)
    combined_video[prompt_frames:requested_duration] = video
    normalized = ((reference - mean) / std).astype(np.float32)
    if not np.isfinite(normalized).all():
        raise ValueError("Non-finite normalized reference")
    return normalized, combined_video, prompt_frames, target_frames, effective_duration

def word_edit_counts(truth: str, hypothesis: str) -> tuple[int, int]:
    """Exact word Levenshtein numerator and reference-word denominator."""
    reference, prediction = truth.split(), hypothesis.split()
    if not reference:
        raise ValueError("WER reference contains no words")
    previous = list(range(len(prediction) + 1))
    for i, word in enumerate(reference, 1):
        current = [i]
        for j, candidate in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1], len(reference)
