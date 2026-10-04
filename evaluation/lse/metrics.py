"""SyncNet LSE scoring, matching the official SyncNet/Wav2Lip equations."""

import numpy as np
import python_speech_features
import torch
import torch.nn.functional as F


def score_embeddings(video_features, audio_features, vshift=15):
    """Mean window distance at each shift, then minimum and median-minus-minimum.

    Keep upstream zero padding and pairwise_distance epsilon. Taking a minimum
    per window before averaging would produce a different (optimistic) metric.
    """
    if video_features.ndim != 2 or video_features.shape != audio_features.shape:
        raise ValueError("Audio/video embeddings must have the same [windows, features] shape")
    if not len(video_features) or vshift < 1:
        raise ValueError("Need at least one window and vshift >= 1")
    if not torch.isfinite(video_features).all() or not torch.isfinite(audio_features).all():
        raise ValueError("Non-finite SyncNet embeddings")
    padded = F.pad(audio_features, (0, 0, vshift, vshift))
    distances = torch.stack(
        [
            F.pairwise_distance(video_features[[i]].repeat(2 * vshift + 1, 1), padded[i : i + 2 * vshift + 1])
            for i in range(len(video_features))
        ],
        dim=1,
    )
    mean_distances = distances.mean(dim=1)
    minimum, index = mean_distances.min(dim=0)
    return {
        "lse_d": minimum.item(),
        "lse_c": (mean_distances.median() - minimum).item(),
        "offset_frames": vshift - index.item(),
        "offset_seconds": (vshift - index.item()) / 25,
        "zero_offset_distance": mean_distances[vshift].item(),
        "distance_by_shift": mean_distances.tolist(),
        "num_windows": len(video_features),
    }


@torch.inference_mode()
def extract_embeddings(model, frames, audio, device, batch_size=20):
    """224px BGR uint8 frames and mono int16 16kHz audio; no normalization."""
    if frames.ndim != 4 or frames.shape[1:] != (224, 224, 3) or frames.dtype != np.uint8:
        raise ValueError("SyncNet expects uint8 BGR frames with shape [T, 224, 224, 3]")
    if audio.ndim != 1 or audio.dtype != np.int16:
        raise ValueError("SyncNet MFCCs must be computed from mono PCM int16 audio")
    # Preserve the upstream final-window convention, including its excluded endpoint.
    lastframe = min(len(frames), len(audio) // 640) - 5
    if lastframe <= 0:
        raise ValueError("Clip is too short: need at least 6 frames / 0.24s of audio")
    mfcc = python_speech_features.mfcc(audio, 16000).T
    if not np.isfinite(mfcc).all():
        raise ValueError("Non-finite MFCCs")
    image_features, audio_features = [], []
    for start in range(0, lastframe, batch_size):
        indices = range(start, min(lastframe, start + batch_size))
        images = np.stack([frames[i : i + 5].transpose(3, 0, 1, 2) for i in indices])
        sounds = np.stack([mfcc[:, i * 4 : i * 4 + 20] for i in indices])[:, None]
        image_features.append(model.forward_lip(torch.from_numpy(images).float().to(device)).cpu())
        audio_features.append(model.forward_aud(torch.from_numpy(sounds).float().to(device)).cpu())
    return torch.cat(image_features), torch.cat(audio_features)
