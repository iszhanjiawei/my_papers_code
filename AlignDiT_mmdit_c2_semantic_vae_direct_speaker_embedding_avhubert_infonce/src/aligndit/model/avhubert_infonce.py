"""Raw text-context alignment to frozen, native-rate AV-HuBERT audio targets.

Each rank averages its valid anchors; DDP then averages rank gradients, matching
the existing trainer's rank-local loss convention. There are no collectives or
cross-clip negatives in this module.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def sample_context_on_teacher_grid(student, student_lengths, generation_mask, teacher_frames):
    """Sample 40 Hz context at fixed 25 Hz frame centres, without clip stretching.

    The coordinates are ``(j + .5) * 40 / 25 - .5`` even when a teacher clip has
    tail padding or shares a batch with longer clips. An anchor is valid only
    when every source position with nonzero interpolation weight is valid and
    belongs to the synthesized region.
    """
    if student.ndim != 3 or student.shape[1] < 1:
        raise ValueError("student must have shape [batch, positive_frames, channels]")
    if generation_mask.dtype != torch.bool or generation_mask.shape != student.shape[:2]:
        raise ValueError("generation_mask must be bool and match the student's batch/frame axes")
    if student_lengths.shape != (student.shape[0],) or student_lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("student_lengths must be an integer [batch] tensor")
    if student_lengths.device != student.device or generation_mask.device != student.device:
        raise ValueError("student lengths and generation mask must be on the student's device")
    if (student_lengths < 0).any() or (student_lengths > student.shape[1]).any():
        raise ValueError("student lengths exceed the padded student sequence")
    if type(teacher_frames) is not int or teacher_frames < 0:
        raise ValueError("teacher_frames must be a nonnegative integer")

    positions = (torch.arange(teacher_frames, device=student.device, dtype=torch.float32) + 0.5) * 1.6 - 0.5
    lower = positions.floor().long()
    upper = positions.ceil().long()
    fraction = positions - lower
    lower_safe = lower.clamp(0, student.shape[1] - 1)
    upper_safe = upper.clamp(0, student.shape[1] - 1)
    samples = torch.lerp(student[:, lower_safe].float(), student[:, upper_safe].float(), fraction[None, :, None])
    valid = (lower[None, :] >= 0) & (upper[None, :] < student_lengths[:, None])
    valid = valid & generation_mask[:, lower_safe] & generation_mask[:, upper_safe]
    return samples, valid


def temporal_context_infonce(
    student,
    teacher,
    student_lengths,
    teacher_lengths,
    teacher_valid_lengths,
    generation_mask,
    *,
    temperature=0.07,
    min_negative_frames=5,
    enabled=True,
):
    """One-way same-clip InfoNCE with the matching teacher frame as positive.

    Teacher keys are its valid prefix on the original 25 Hz grid. Negatives
    satisfy ``abs(key - anchor) >= min_negative_frames``; the diagonal remains
    positive. Rows without any negative are excluded. Empty/disabled batches
    return a graph-connected zero so DDP still sees the student's projector.
    """
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if type(min_negative_frames) is not int or min_negative_frames < 1:
        raise ValueError("min_negative_frames must be a positive integer")
    if student.ndim != 3 or teacher.ndim != 3 or student.shape[0] != teacher.shape[0]:
        raise ValueError("student and teacher must have matching [batch, frames, channels] tensors")
    if student.shape[2] != teacher.shape[2] or student.device != teacher.device:
        raise ValueError("student and teacher must have matching channels and device")
    if not student.is_floating_point() or not teacher.is_floating_point():
        raise TypeError("student and teacher must be floating-point features")
    for lengths in (teacher_lengths, teacher_valid_lengths):
        if lengths.shape != (teacher.shape[0],) or lengths.dtype not in (torch.int32, torch.int64):
            raise ValueError("teacher lengths must be integer [batch] tensors")
        if lengths.device != student.device:
            raise ValueError("teacher lengths must be on the student's device")
    if (teacher_lengths < 0).any() or (teacher_lengths > teacher.shape[1]).any():
        raise ValueError("teacher_lengths exceed its padded sequence")
    if (teacher_valid_lengths < 0).any() or (teacher_valid_lengths > teacher_lengths).any():
        raise ValueError("teacher_valid_lengths must be within the original teacher lengths")

    stats = {
        "infonce_valid_anchors": 0,
        "infonce_positive_similarity": 0.0,
        "infonce_negative_similarity": 0.0,
        "infonce_retrieval_accuracy": 0.0,
        "infonce_negative_pairs": 0,
        "infonce_negatives_per_anchor": 0.0,
    }
    loss_sum = student[..., :0].float().sum()
    if not enabled:
        return loss_sum, stats

    with torch.autocast(device_type=student.device.type, enabled=False):
        sampled, eligible = sample_context_on_teacher_grid(
            student, student_lengths, generation_mask, teacher.shape[1]
        )
        positive_sum = student.new_zeros((), dtype=torch.float32)
        negative_sum = student.new_zeros((), dtype=torch.float32)
        correct_sum = student.new_zeros((), dtype=torch.float32)
        negative_count = 0
        for batch_index, valid_length in enumerate(teacher_valid_lengths.tolist()):
            if valid_length <= min_negative_frames:
                continue
            anchor_indices = eligible[batch_index, :valid_length].nonzero(as_tuple=False).flatten()
            if anchor_indices.numel() == 0:
                continue
            key_indices = torch.arange(valid_length, device=student.device)
            negatives = (anchor_indices[:, None] - key_indices[None, :]).abs() >= min_negative_frames
            has_negative = negatives.any(dim=1)
            anchor_indices = anchor_indices[has_negative]
            negatives = negatives[has_negative]
            if anchor_indices.numel() == 0:
                continue
            queries = F.normalize(sampled[batch_index, anchor_indices], dim=-1, eps=1e-8)
            keys = F.normalize(teacher[batch_index, :valid_length].detach().float(), dim=-1, eps=1e-8)
            similarity = queries @ keys.transpose(0, 1)
            positives = anchor_indices[:, None] == key_indices[None, :]
            logits = (similarity / temperature).masked_fill(~(positives | negatives), -torch.inf)
            loss_sum = loss_sum + F.cross_entropy(logits, anchor_indices, reduction="sum")
            count = anchor_indices.numel()
            stats["infonce_valid_anchors"] += count
            positive_sum = positive_sum + similarity.detach()[positives].sum()
            negative_sum = negative_sum + similarity.detach()[negatives].sum()
            negative_count += int(negatives.sum().item())
            correct_sum = correct_sum + (logits.detach().argmax(-1) == anchor_indices).sum()

        anchor_count = stats["infonce_valid_anchors"]
        if anchor_count:
            stats["infonce_negative_pairs"] = negative_count
            stats["infonce_negatives_per_anchor"] = negative_count / anchor_count
            stats["infonce_positive_similarity"] = float(positive_sum.item() / anchor_count)
            stats["infonce_negative_similarity"] = float(negative_sum.item() / negative_count)
            stats["infonce_retrieval_accuracy"] = float(correct_sum.item() / anchor_count)
            return loss_sum / anchor_count, stats
        return loss_sum, stats
