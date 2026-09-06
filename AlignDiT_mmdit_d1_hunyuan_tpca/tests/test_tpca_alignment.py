"""Numerical and information-flow checks for the standalone TPCA aligner."""

import importlib.util
import itertools
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/aligndit/model/tpca.py"
SPEC = importlib.util.spec_from_file_location("tpca_standalone", MODULE_PATH)
tpca = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tpca)
OccurrenceCTCAligner = tpca.OccurrenceCTCAligner


def enumerate_state_posteriors(log_probs, tokens, blank):
    """Independent enumeration of legal CTC state paths for tiny examples."""
    labels = [blank]
    for token in tokens:
        labels.extend([token, blank])
    steps, states = log_probs.size(0), len(labels)
    posterior = torch.zeros(steps, states, dtype=log_probs.dtype)
    total = 0.0
    for path in itertools.product(range(states), repeat=steps):
        if path[0] not in (0, 1) or path[-1] not in (states - 2, states - 1):
            continue
        valid = True
        for previous, current in itertools.pairwise(path):
            delta = current - previous
            if delta not in (0, 1, 2) or (
                delta == 2 and (labels[current] == blank or labels[current] == labels[previous])
            ):
                valid = False
                break
        if not valid:
            continue
        weight = sum(log_probs[t, labels[state]] for t, state in enumerate(path)).exp()
        total += weight
        for t, state in enumerate(path):
            posterior[t, state] += weight
    return posterior / total


def test_forward_backward_matches_exhaustive_paths_and_variable_lengths():
    torch.manual_seed(11)
    log_probs = torch.randn(2, 4, 3).log_softmax(-1)
    targets = torch.tensor([[0, 1], [0, 0]])
    lengths = torch.tensor([4, 3])
    result = tpca._ctc_state_posteriors(log_probs, targets, lengths, torch.tensor([2, 2]), 2)
    for b, tokens in enumerate(([0, 1], [0, 0])):
        expected = enumerate_state_posteriors(log_probs[b, :lengths[b]], tokens, 2)
        torch.testing.assert_close(result[b, :lengths[b]], expected, atol=2e-6, rtol=2e-6)
    assert result[1, 3].count_nonzero() == 0


def test_variable_transcript_lengths_mask_extra_occurrence_states():
    torch.manual_seed(18)
    log_probs = torch.randn(2, 4, 3).log_softmax(-1)
    targets = torch.tensor([[0, 1], [1, 0]])
    result = tpca._ctc_state_posteriors(log_probs, targets, torch.tensor([4, 3]), torch.tensor([2, 1]), 2)
    expected = enumerate_state_posteriors(log_probs[1, :3], [1], 2)
    torch.testing.assert_close(result[1, :3, :3], expected, atol=2e-6, rtol=2e-6)
    assert result[1, :, 3:].count_nonzero() == 0
    assert result[1, 3].count_nonzero() == 0


def test_repeated_tokens_have_distinct_occurrence_columns():
    # AA with three frames has exactly one legal path: A, blank, A.
    aligner = OccurrenceCTCAligner(4, 2, upsample_factor=1)
    result = aligner(torch.randn(1, 3, 4), torch.tensor([[0, 0]]), torch.ones(1, 3, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
    expected = torch.tensor([[[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]])
    torch.testing.assert_close(result["prior"], expected)
    assert not result["prior"].requires_grad


def test_padding_values_and_padded_length_do_not_affect_valid_output():
    torch.manual_seed(12)
    aligner = OccurrenceCTCAligner(4, 3)
    visual = torch.randn(1, 4, 4)
    text = torch.tensor([[0, 1, 0]])
    original = aligner(visual, text, torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 3, dtype=torch.bool))
    padded_visual = torch.cat([visual, torch.full((1, 5, 4), 999.)], 1)
    padded_text = torch.tensor([[0, 1, 0, -1, -1]])
    padded = aligner(padded_visual, padded_text, torch.arange(9)[None] < 4, padded_text != -1)
    torch.testing.assert_close(padded["prior"][:, :4, :3], original["prior"][:, :, :3])
    torch.testing.assert_close(padded["prior"][:, :4, -1], original["prior"][:, :, -1])
    torch.testing.assert_close(padded["ctc_loss"], original["ctc_loss"])
    assert padded["prior"][:, :, 3:5].count_nonzero() == 0
    assert torch.equal(padded["prior"][:, 4:, -1], torch.ones(1, 5))


def test_empty_and_infeasible_samples_are_null_and_excluded_from_loss():
    torch.manual_seed(13)
    aligner = OccurrenceCTCAligner(4, 2, upsample_factor=1)
    video = torch.randn(4, 3, 4, requires_grad=True)
    text = torch.tensor([[0, 1], [0, 0], [-1, -1], [0, 1]])
    video_mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 1, 1], [0, 0, 0]], dtype=torch.bool)
    result = aligner(video, text, video_mask, text != -1)
    single = aligner(video[:1], text[:1], video_mask[:1], text[:1] != -1)
    torch.testing.assert_close(result["ctc_loss"], single["ctc_loss"])
    assert result["feasible_fraction"].item() == 0.25
    assert result["feasible_mask"].tolist() == [True, False, False, False]
    assert result["prior"][1:, :, :-1].count_nonzero() == 0
    assert torch.equal(result["prior"][1:, :, -1], torch.ones(3, 3))
    result["ctc_loss"].backward()
    assert torch.isfinite(video.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in aligner.parameters())
    assert video.grad[1:].count_nonzero() == 0
    for video_length, text_length in [(0, 2), (3, 0), (0, 0)]:
        empty = aligner(torch.zeros(1, video_length, 4), torch.zeros(1, text_length, dtype=torch.long), torch.ones(1, video_length, dtype=torch.bool), torch.ones(1, text_length, dtype=torch.bool))
        assert empty["prior"].shape == (1, video_length, text_length + 1)
        assert torch.isfinite(empty["ctc_loss"])
        empty["ctc_loss"].backward()


def test_teacher_free_head_does_not_receive_or_depend_on_text():
    torch.manual_seed(14)
    aligner = OccurrenceCTCAligner(4, 3)
    video = torch.randn(1, 4, 4)
    observed = []
    handle = aligner.output_proj.register_forward_hook(lambda module, args, output: observed.append(output.detach().clone()))
    masks = torch.ones(1, 4, dtype=torch.bool)
    for text in [torch.tensor([[0, 1]]), torch.tensor([[2, 2]])]:
        result = aligner(video, text, masks, torch.ones_like(text, dtype=torch.bool), compute_loss=False)
        assert result["ctc_loss"].item() == 0
    handle.remove()
    torch.testing.assert_close(observed[0], observed[1], atol=0, rtol=0)


def test_span_mapping_matches_independent_target_segment_and_blocks_prompt():
    torch.manual_seed(15)
    aligner = OccurrenceCTCAligner(4, 3)
    video = torch.randn(1, 7, 4)
    text = torch.tensor([[2, 0, 1, 2]])
    full = aligner(video, text, torch.ones(1, 7, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool), video_start=torch.tensor([2]), video_end=torch.tensor([6]), text_start=torch.tensor([1]), text_end=torch.tensor([3]))
    target = aligner(video[:, 2:6], text[:, 1:3], torch.ones(1, 4, dtype=torch.bool), torch.ones(1, 2, dtype=torch.bool))
    torch.testing.assert_close(full["prior"][:, 2:6, 1:3], target["prior"][:, :, :2])
    torch.testing.assert_close(full["prior"][:, 2:6, -1], target["prior"][:, :, -1])
    torch.testing.assert_close(full["ctc_loss"], target["ctc_loss"])
    assert full["prior"][:, :, [0, 3]].count_nonzero() == 0
    assert torch.equal(full["prior"][:, [0, 1, 6], -1], torch.ones(1, 3))
    changed = video.clone()
    changed[:, :2] = 1e5
    changed[:, 6:] = -1e5
    other = aligner(changed, text, torch.ones(1, 7, dtype=torch.bool), torch.ones(1, 4, dtype=torch.bool), video_start=torch.tensor([2]), video_end=torch.tensor([6]), text_start=torch.tensor([1]), text_end=torch.tensor([3]))
    torch.testing.assert_close(full["prior"], other["prior"])


def test_upsampled_posterior_is_averaged_to_native_video_clock():
    torch.manual_seed(16)
    aligner = OccurrenceCTCAligner(4, 3, upsample_factor=2)
    video = torch.randn(1, 3, 4)
    text = torch.tensor([[0, 1]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    result = aligner(video, text, mask, torch.ones_like(text, dtype=torch.bool))
    log_probs = aligner.compute_logits(video, mask).log_softmax(-1)
    posterior = tpca._ctc_state_posteriors(log_probs.detach(), text, torch.tensor([6]), torch.tensor([2]), 3)
    expected = posterior[:, :, 1::2].reshape(1, 3, 2, 2).mean(2)
    torch.testing.assert_close(result["prior"][:, :, :-1], expected)
    torch.testing.assert_close(result["prior"].sum(-1), torch.ones(1, 3))


def test_independent_subframes_can_emit_two_tokens_and_both_receive_gradients():
    torch.manual_seed(19)
    aligner = OccurrenceCTCAligner(4, 2, upsample_factor=2)
    with torch.no_grad():
        aligner.output_proj.weight.zero_()
        bias = aligner.output_proj.bias.reshape(2, 3)
        bias.zero_()
        bias[0, 0] = 3.0
        bias[1, 1] = 3.0
    video = torch.randn(1, 1, 4)
    video_mask = torch.ones(1, 1, dtype=torch.bool)
    logits = aligner.compute_logits(video, video_mask)
    assert logits.shape == (1, 2, 3)
    assert logits.argmax(-1).tolist() == [[0, 1]]
    assert not torch.equal(logits[:, 0], logits[:, 1])
    result = aligner(video, torch.tensor([[0, 1]]), video_mask, torch.ones(1, 2, dtype=torch.bool))
    assert result["feasible_mask"].tolist() == [True]
    # One native visual frame carries both independently predicted emissions;
    # the returned prior still lives on the original visual clock.
    torch.testing.assert_close(result["prior"], torch.tensor([[[0.5, 0.5, 0.0]]]))
    result["ctc_loss"].backward()
    subframe_grads = aligner.output_proj.weight.grad.reshape(2, 3, -1)
    assert torch.isfinite(subframe_grads).all()
    assert bool((subframe_grads.abs().sum(dim=(1, 2)) > 0).all())
    assert bool((aligner.output_proj.bias.grad.reshape(2, 3).abs().sum(-1) > 0).all())


if __name__ == "__main__":
    # The project environment does not require pytest; retain pytest discovery
    # compatibility while allowing this focused check to run directly.
    tests = [(name, fn) for name, fn in globals().copy().items() if name.startswith("test_") and callable(fn)]
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"{len(tests)} tests passed")
