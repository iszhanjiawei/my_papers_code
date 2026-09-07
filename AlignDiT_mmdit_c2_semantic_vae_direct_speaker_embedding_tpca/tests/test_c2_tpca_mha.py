"""C2-specific MHA preservation tests for the TPCA adaptation."""
import unittest

import torch
from torch import nn

from aligndit.model.tpca_attention import apply_tpca_multihead_attention


def _inputs(text_dim=12, dropout=0.0):
    torch.manual_seed(317)
    attention = nn.MultiheadAttention(16, 4, dropout=dropout, kdim=text_dim, vdim=text_dim, batch_first=True)
    audio = torch.randn(2, 6, 16)
    text = torch.randn(2, 4, text_dim)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    posterior = torch.rand(2, 5, 5)
    posterior[0, :, 3] = 0
    posterior /= posterior.sum(-1, keepdim=True)
    context = dict(
        local_heads=2, query_chunk_size=2, bias_strength=1.0,
        query_mask=torch.tensor([[False, False, True, True, True, True], [False, False, True, True, True, True]]),
        text_mask=mask, video_mask=torch.ones(2, 5, dtype=torch.bool),
        posterior=posterior, prior_smoothing=0.1,
        audio_query=torch.randn(2, 4, 6, 4, requires_grad=True),
        video_key=torch.randn(2, 4, 5, 4, requires_grad=True),
    )
    return attention, audio, text, mask, context


class TestC2TPCAMHA(unittest.TestCase):
    def test_disabled_strength_matches_original_mha_for_separate_and_combined_projections(self):
        for text_dim in (12, 16):
            attention, audio, text, mask, context = _inputs(text_dim)
            attention.eval()
            context['bias_strength'] = 0.0
            reference = attention(audio, text, text, key_padding_mask=~mask, need_weights=False)[0]
            output, loss = apply_tpca_multihead_attention(attention, audio, text, mask, context)
            torch.testing.assert_close(output, reference, atol=1e-7, rtol=1e-6)
            assert loss.item() == 0


    def test_tpca_keeps_prompt_queries_and_uses_existing_projections(self):
        attention, audio, text, mask, context = _inputs()
        attention.eval()
        reference = attention(audio, text, text, key_padding_mask=~mask, need_weights=False)[0]
        original_keys = set(attention.state_dict())
        output, loss = apply_tpca_multihead_attention(attention, audio, text, mask, context)
        torch.testing.assert_close(output[:, :2], reference[:, :2], atol=1e-7, rtol=1e-6)
        assert not torch.allclose(output[:, 2:], reference[:, 2:])
        assert loss.item() > 0
        assert set(attention.state_dict()) == original_keys
        (output.square().mean() + loss).backward()
        for name, parameter in attention.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        assert context['audio_query'].grad is None
        assert context['video_key'].grad is None


    def test_all_inactive_cfg_rows_equal_original_mha_and_have_zero_loss(self):
        attention, audio, text, mask, context = _inputs()
        attention.eval()
        context['query_mask'].zero_()
        reference = attention(audio, text, text, key_padding_mask=~mask, need_weights=False)[0]
        output, loss = apply_tpca_multihead_attention(attention, audio, text, mask, context)
        torch.testing.assert_close(output, reference, atol=1e-7, rtol=1e-6)
        assert loss.item() == 0


    def test_original_attention_dropout_is_retained_on_tpca_heads(self):
        attention, audio, text, mask, context = _inputs(dropout=1.0)
        attention.train()
        output, loss = apply_tpca_multihead_attention(attention, audio, text, mask, context)
        expected = attention.out_proj.bias[None, None].expand_as(output)
        torch.testing.assert_close(output, expected)
        assert torch.isfinite(loss) and loss.item() > 0


if __name__ == '__main__':
    unittest.main()
