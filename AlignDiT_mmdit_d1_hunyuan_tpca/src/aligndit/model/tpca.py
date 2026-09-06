"""Visual-only CTC alignment over transcript *occurrences* for TPCA.

The recognition head never sees text.  Text is used only to condition the CTC
path distribution.  Its blank states are non-emissions, not silence labels;
the returned posterior is not a phoneme-duration annotation.
"""


import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


@torch.jit.script
def _ctc_state_posteriors(
    log_probs: Tensor,
    targets: Tensor,
    input_lengths: Tensor,
    target_lengths: Tensor,
    blank: int,
) -> Tensor:
    """Log-space forward/backward; inputs are feasible, nonempty sequences.

    Args: log_probs (B,T,C), targets (B,L), lengths (B,).
    Returns: (B,T,2*L+1), including distinct blank and token-occurrence states.
    Padding states/timesteps have zero posterior.  Call on detached FP32 input.
    """
    batch, steps, _ = log_probs.shape
    max_tokens = targets.size(1)
    states = 2 * max_tokens + 1
    state_ids = torch.arange(states, device=log_probs.device)
    labels = torch.full((batch, states), blank, dtype=torch.long, device=log_probs.device)
    labels[:, 1::2] = targets
    state_valid = state_ids.unsqueeze(0) <= 2 * target_lengths.unsqueeze(1)
    emissions = log_probs.gather(2, labels.unsqueeze(1).expand(batch, steps, states))
    emissions = emissions.masked_fill(~state_valid.unsqueeze(1), float("-inf"))
    skip = torch.zeros((batch, states), dtype=torch.bool, device=log_probs.device)
    skip[:, 2:] = (labels[:, 2:] != blank) & (labels[:, 2:] != labels[:, :-2])
    skip = skip & state_valid
    neg_one = torch.full((batch, 1), float("-inf"), device=log_probs.device, dtype=log_probs.dtype)
    neg_two = torch.full((batch, 2), float("-inf"), device=log_probs.device, dtype=log_probs.dtype)

    alpha = torch.full_like(emissions, float("-inf"))
    current = torch.full((batch, states), float("-inf"), device=log_probs.device, dtype=log_probs.dtype)
    current[:, :2] = emissions[:, 0, :2]
    alpha[:, 0] = current
    for t in range(1, steps):
        previous_one = torch.cat((neg_one, current[:, :-1]), dim=1)
        previous_two = torch.cat((neg_two, current[:, :-2]), dim=1).masked_fill(~skip, float("-inf"))
        current = emissions[:, t] + torch.logaddexp(torch.logaddexp(current, previous_one), previous_two)
        current = current.masked_fill((t >= input_lengths).unsqueeze(1), float("-inf"))
        alpha[:, t] = current

    batch_ids = torch.arange(batch, device=log_probs.device)
    final_alpha = alpha[batch_ids, input_lengths - 1]
    final_blank = final_alpha.gather(1, (2 * target_lengths).unsqueeze(1)).squeeze(1)
    final_token = final_alpha.gather(1, (2 * target_lengths - 1).unsqueeze(1)).squeeze(1)
    log_z = torch.logaddexp(final_blank, final_token)

    beta = torch.full_like(emissions, float("-inf"))
    current = torch.full_like(current, float("-inf"))
    terminal = (state_ids.unsqueeze(0) == 2 * target_lengths.unsqueeze(1)) | (
        state_ids.unsqueeze(0) == 2 * target_lengths.unsqueeze(1) - 1
    )
    for rev_t in range(steps):
        t = steps - 1 - rev_t
        if t < steps - 1:
            next_emission = current + emissions[:, t + 1]
            following_one = torch.cat((next_emission[:, 1:], neg_one), dim=1)
            following_two = torch.cat((next_emission[:, 2:].masked_fill(~skip[:, 2:], float("-inf")), neg_two), dim=1)
            current = torch.logaddexp(torch.logaddexp(next_emission, following_one), following_two)
        current = torch.where(
            ((t == input_lengths - 1).unsqueeze(1) & terminal),
            torch.zeros_like(current),
            current,
        )
        current = current.masked_fill(~state_valid | (t >= input_lengths).unsqueeze(1), float("-inf"))
        beta[:, t] = current

    posterior = (alpha + beta - log_z[:, None, None]).exp()
    # Normalization removes accumulated FP32 rounding, without assigning mass
    # to invalid timesteps (whose row sum remains zero).
    return posterior / posterior.sum(dim=-1, keepdim=True).clamp_min(1e-20)


class OccurrenceCTCAligner(nn.Module):
    """Predict a detached visual-to-transcript occurrence prior.

    Raw text IDs are ``0 .. vocab_size-1``; ``-1`` is padding and
    ``vocab_size`` is the internal CTC blank.  Masks are True for valid entries.
    Optional start/end tensors use original, half-open sequence coordinates.
    Only selected video/text positions participate in alignment and its loss.
    Outside those positions, including padding, the prior is entirely null.
    ``feasible_mask`` identifies samples eligible for downstream path losses;
    exclude False entries even though their null prior is numerically valid.
    """

    def __init__(
        self,
        video_dim: int,
        vocab_size: int,
        upsample_factor: int = 2,
        hidden_dim: int | None = None,
        temporal_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if video_dim <= 0 or vocab_size <= 0 or upsample_factor < 1:
            raise ValueError("video_dim, vocab_size and upsample_factor must be positive")
        if temporal_kernel_size < 1 or temporal_kernel_size % 2 != 1:
            raise ValueError("temporal_kernel_size must be a positive odd integer")
        hidden_dim = min(video_dim, 256) if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.video_dim = video_dim
        self.vocab_size = vocab_size
        self.blank = vocab_size
        self.upsample_factor = upsample_factor
        self.input_norm = nn.LayerNorm(video_dim)
        self.input_proj = nn.Linear(video_dim, hidden_dim)
        self.temporal_conv = nn.Conv1d(
            hidden_dim, hidden_dim, temporal_kernel_size,
            padding=temporal_kernel_size // 2, groups=hidden_dim,
        )
        # Each native frame predicts independently parameterized subframe
        # emissions.  Repeating one logit would leave greedy CTC decoding
        # unable to emit more than one token per native video frame.
        self.output_proj = nn.Linear(hidden_dim, upsample_factor * (vocab_size + 1))

    def compute_logits(self, video: Tensor, video_mask: Tensor) -> Tensor:
        """Return (B,V*factor,vocab+blank) visual-only CTC emissions.

        Subframes have distinct classifier parameters, but share the native
        visual representation: a finer prediction grid adds no visual evidence.
        """
        if video.size(1) == 0:
            return video.new_zeros((video.size(0), 0, self.vocab_size + 1)) + self.output_proj.weight.sum() * 0
        # Mask before every temporal operation: padded values and LayerNorm/
        # projection biases must not change valid boundary-frame predictions.
        visual = video.masked_fill(~video_mask.unsqueeze(-1), 0)
        hidden = F.gelu(self.input_proj(self.input_norm(visual)))
        hidden = hidden.masked_fill(~video_mask.unsqueeze(-1), 0)
        hidden = self.temporal_conv(hidden.transpose(1, 2)).transpose(1, 2)
        logits = self.output_proj(F.gelu(hidden)).reshape(
            video.size(0), video.size(1) * self.upsample_factor, self.vocab_size + 1,
        )
        emission_mask = video_mask.repeat_interleave(self.upsample_factor, dim=1)
        return logits.masked_fill(~emission_mask.unsqueeze(-1), 0)

    @staticmethod
    def _span_mask(mask: Tensor, start: Tensor | None, end: Tensor | None) -> Tensor:
        batch, length = mask.shape
        if start is None and end is None:
            return mask.bool()
        start = torch.zeros(batch, device=mask.device, dtype=torch.long) if start is None else start.to(mask.device)
        end = torch.full((batch,), length, device=mask.device, dtype=torch.long) if end is None else end.to(mask.device)
        if start.shape != (batch,) or end.shape != (batch,):
            raise ValueError("span start/end must have shape (batch,)")
        if start.is_floating_point() or end.is_floating_point():
            raise ValueError("span start/end must be integer coordinates")
        if bool(((start < 0) | (end > length) | (start > end)).any()):
            raise ValueError("invalid half-open span")
        positions = torch.arange(length, device=mask.device).unsqueeze(0)
        return mask.bool() & (positions >= start.unsqueeze(1)) & (positions < end.unsqueeze(1))

    def forward(
        self,
        video: Tensor,
        text: Tensor,
        video_mask: Tensor,
        text_mask: Tensor,
        video_start: Tensor | None = None,
        video_end: Tensor | None = None,
        text_start: Tensor | None = None,
        text_end: Tensor | None = None,
        compute_loss: bool = True,
    ) -> dict[str, Tensor]:
        if video.ndim != 3 or text.ndim != 2 or video.size(0) != text.size(0):
            raise ValueError("expected video (B,V,D) and text (B,L)")
        if video.size(-1) != self.video_dim:
            raise ValueError("video feature dimension does not match video_dim")
        if video_mask.shape != video.shape[:2] or text_mask.shape != text.shape:
            raise ValueError("padding-mask shapes must match their sequences")
        if text.dtype not in (torch.int32, torch.int64):
            raise ValueError("raw text IDs must be integer tensors")
        if video.device != text.device or video.device != video_mask.device or video.device != text_mask.device:
            raise ValueError("video, text and masks must be on the same device")
        selected_video = self._span_mask(video_mask, video_start, video_end)
        selected_text = self._span_mask(text_mask, text_start, text_end) & text.ne(-1)
        if bool((selected_text & ((text < 0) | (text >= self.vocab_size))).any()):
            raise ValueError("valid raw text IDs must be in [0, vocab_size)")

        # Crucially, neither text nor its embeddings enter this call.
        logits = self.compute_logits(video, selected_video)
        batch, native_steps = video.shape[:2]
        text_steps = text.size(1)
        prior = torch.zeros((batch, native_steps, text_steps + 1), device=video.device, dtype=torch.float32)
        prior[..., -1] = 1
        zero_loss = logits.sum() * 0
        video_indices = [selected_video[b].nonzero(as_tuple=False).flatten() for b in range(batch)]
        text_indices = [selected_text[b].nonzero(as_tuple=False).flatten() for b in range(batch)]
        targets = [text[b].index_select(0, indices).long() for b, indices in enumerate(text_indices)]
        # A repeated token requires a separating blank in CTC.  Compute this
        # exact feasibility condition before either native CTC or posterior DP.
        repeats = torch.stack([(target[1:] == target[:-1]).sum() for target in targets]) if batch else text.new_zeros(0)
        repeats_cpu = repeats.detach().cpu().tolist()
        feasible = [
            b for b in range(batch)
            if targets[b].numel() > 0
            and video_indices[b].numel() * self.upsample_factor >= targets[b].numel() + repeats_cpu[b]
        ]
        feasible_fraction = prior.new_tensor(len(feasible) / max(batch, 1))
        feasible_mask = torch.zeros(batch, dtype=torch.bool, device=video.device)
        feasible_mask[feasible] = True
        if not feasible:
            return {
                "prior": prior, "ctc_loss": zero_loss,
                "feasible_fraction": feasible_fraction, "feasible_mask": feasible_mask,
            }

        selected_logits = [
            logits[b].reshape(native_steps, self.upsample_factor, self.vocab_size + 1)
            .index_select(0, video_indices[b]).reshape(-1, self.vocab_size + 1)
            for b in feasible
        ]
        selected_targets = [targets[b] for b in feasible]
        log_probs = pad_sequence(selected_logits, batch_first=True).float().log_softmax(dim=-1)
        target_padded = pad_sequence(selected_targets, batch_first=True, padding_value=0)
        input_lengths = torch.tensor([x.size(0) for x in selected_logits], dtype=torch.long, device=video.device)
        target_lengths = torch.tensor([x.numel() for x in selected_targets], dtype=torch.long, device=video.device)
        ctc_loss = zero_loss
        if compute_loss:
            per_sample = F.ctc_loss(
                log_probs.transpose(0, 1), torch.cat(selected_targets), input_lengths,
                target_lengths, blank=self.blank, reduction="none", zero_infinity=True,
            )
            ctc_loss = (per_sample / target_lengths).mean()

        with torch.no_grad():
            posterior = _ctc_state_posteriors(
                log_probs.detach(), target_padded, input_lengths, target_lengths, self.blank,
            )
            for packed_b, original_b in enumerate(feasible):
                count_v = video_indices[original_b].numel()
                count_t = targets[original_b].numel()
                occurrence = posterior[packed_b, :count_v * self.upsample_factor, 1:2 * count_t:2]
                native_occurrence = occurrence.reshape(count_v, self.upsample_factor, count_t).mean(dim=1)
                blank_mass = posterior[packed_b, :count_v * self.upsample_factor, 0:2 * count_t + 1:2].sum(dim=-1)
                native_null = blank_mass.reshape(count_v, self.upsample_factor).mean(dim=1)
                rows = video_indices[original_b]
                columns = text_indices[original_b]
                prior[original_b, rows[:, None], columns[None, :]] = native_occurrence
                prior[original_b, rows, -1] = native_null
        return {
            "prior": prior.detach(), "ctc_loss": ctc_loss,
            "feasible_fraction": feasible_fraction, "feasible_mask": feasible_mask,
        }
