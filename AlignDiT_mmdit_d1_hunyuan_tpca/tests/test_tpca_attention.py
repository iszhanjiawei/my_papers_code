"""Numerical and integrated TPCA regression checks; no data or training jobs."""
import copy
import math
import unittest

import torch
import torch.nn.functional as F

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, HunyuanDualTextCrossAttention
from aligndit.model.tpca_attention import apply_tpca_audio_attention, compose_path_prior


def tiny_model():
    return DiT_VT_MMDiT(
        dim=64, depth=3, heads=4, dim_head=16, dropout=0., ff_mult=2,
        mel_dim=8, text_num_embeds=16, text_dim=32, conv_layers=0,
        text_mask_padding=False, qk_norm="rms_norm", pe_attn_head=None,
        attn_mask_enabled=True, use_conformer=False, layer_indices_ctc=[1],
        n_mm_layers=2, n_text_layers=2, prompt_isolated_ca=False,
        audio_video_ratio=4, video_dim=16, text_attention_mode="hunyuan_dual",
        tpca_enabled=True, tpca_layers=[0, 1], tpca_local_heads=2,
        tpca_visual_hidden_dim=16, tpca_warmup_steps=2, tpca_ramp_steps=2,
        tpca_query_chunk_size=5,
    )


def inputs():
    mask = torch.arange(24)[None] < torch.tensor([24, 20])[:, None]
    video_mask = mask[:, ::4]
    text = torch.tensor([[1, 1, 2, 3], [2, 3, 2, -1]])
    generation = mask.clone()
    generation[:, :4] = False
    complement = torch.zeros_like(video_mask)
    complement[:, 0] = True
    return {"x": torch.randn(2, 24, 8), "cond": torch.randn(2, 24, 8),
                "video": torch.randn(2, 6, 16), "text": text, "time": torch.tensor([.2, .7]),
                "mask": mask, "video_mask": video_mask, "text_mask": text != -1,
                "generation_mask": generation, "complementary_mask": complement}


def open_gates(model):
    with torch.no_grad():
        torch.nn.init.normal_(model.proj_out.weight, std=.1)
        for b in model.transformer_blocks:
            b.attn_norm.linear.bias[128:192].fill_(.2)
            if hasattr(b, "v_attn_norm"):
                b.v_attn_norm.linear.bias[128:192].fill_(.2)
                b.cross_attn_ada.bias[128:].fill_(.3)
                b.v_cross_attn_ada.bias[128:].fill_(.3)


class TPCAAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(46)

    def test_path_is_conditional_subblock_of_actual_joint_attention(self):
        q = torch.randn(2, 2, 5, 8)
        ka = torch.randn(2, 2, 5, 8)
        kv = torch.randn(2, 2, 3, 8)
        vm = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
        tm = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
        p = torch.rand(2, 3, 5)
        p[1, :, 2:4] = 0
        p /= p.sum(-1, keepdim=True)
        full = (q @ torch.cat((ka, kv), dim=2).transpose(-1, -2) / math.sqrt(8)).softmax(-1)
        r = full[..., 5:].masked_fill(~vm[:, None, None], 0)
        r /= r.sum(-1, keepdim=True)
        expected = r @ p[:, None]
        actual = compose_path_prior(q, kv, p, vm, tm, 0.)
        torch.testing.assert_close(actual, expected)

    def test_raw_kl_gradients_and_global_prompt_preservation(self):
        q = torch.randn(2, 4, 7, 8, requires_grad=True)
        k = torch.randn(2, 4, 5, 8, requires_grad=True)
        v = torch.randn(2, 4, 5, 8, requires_grad=True)
        qa = torch.randn(2, 4, 7, 8, requires_grad=True)
        kv = torch.randn(2, 4, 3, 8, requires_grad=True)
        p = torch.rand(2, 3, 6, requires_grad=True)
        p = p / p.sum(-1, keepdim=True)
        active = torch.ones(2, 7, dtype=torch.bool)
        active[:, :2] = False
        active[1] = False
        ctx = {"local_heads": 2, "query_chunk_size": 3, "query_mask": active,
                   "audio_query": qa, "video_key": kv, "posterior": p, "bias_strength": .8,
                   "prior_smoothing": .1, "video_mask": torch.ones(2, 3, dtype=torch.bool),
                   "text_mask": torch.ones(2, 5, dtype=torch.bool)}
        base = F.scaled_dot_product_attention(q, k, v)
        out, kl = apply_tpca_audio_attention(q, k, v, base, ctx)
        torch.testing.assert_close(out[:, 2:], base[:, 2:], rtol=0, atol=0)
        torch.testing.assert_close(out[:, :, :2], base[:, :, :2], rtol=0, atol=0)
        torch.testing.assert_close(out[1], base[1], rtol=0, atol=0)
        prior = compose_path_prior(qa[:, :2], kv[:, :2], p, ctx["video_mask"], ctx["text_mask"], .1)
        logits = q[:, :2] @ k[:, :2].transpose(-1, -2) / math.sqrt(8)
        raw = torch.cat((logits, torch.zeros_like(logits[..., :1])), -1).log_softmax(-1)
        expected_kl = (prior * (prior.log() - raw)).sum(-1)[0, :, 2:].mean()
        torch.testing.assert_close(kl, expected_kl)
        grads = torch.autograd.grad(kl, (q, k, qa, kv), allow_unused=True)
        assert grads[0].abs().sum() > 0 and grads[1].abs().sum() > 0
        assert grads[2] is None and grads[3] is None
        torch.testing.assert_close(grads[0][:, 2:], torch.zeros_like(grads[0][:, 2:]))
        zero, loss_zero = apply_tpca_audio_attention(q, k, v, base, {**ctx, "bias_strength": 0.})
        torch.testing.assert_close(zero, base, rtol=0, atol=0)
        assert loss_zero == 0

    def test_warmup_and_condition_drop_match_disabled(self):
        model = tiny_model().eval()
        open_gates(model)
        disabled = copy.deepcopy(model)
        disabled.tpca_enabled = False
        args = inputs()
        with torch.no_grad():
            torch.testing.assert_close(model(**args)[0], disabled(**args)[0], atol=0, rtol=0)
            model.set_tpca_step(4)
            for drops in ({"drop_text": True}, {"drop_video": True}, {"drop_text": True, "drop_video": True}):
                out, aux = model(**args, **drops)
                torch.testing.assert_close(out, disabled(**args, **drops)[0], atol=0, rtol=0)
                assert aux["__tpca__"]["path_loss"] == 0

    def test_packed_cfg_matches_separate_branches_for_batch_two(self):
        model = tiny_model().eval()
        model.set_tpca_step(4)
        open_gates(model)
        args = inputs()
        with torch.no_grad():
            full = model(**args)[0]
            tts = model(**args, drop_video=True)[0]
            null = model(**args, drop_video=True, drop_text=True, drop_audio_cond=True)[0]
            packed = model(**args, cfg_infer=True)[0]
        torch.testing.assert_close(packed, torch.cat((full, tts, null)), atol=2e-6, rtol=2e-5)

    def test_active_gradients_and_activation_checkpoint_equivalence(self):
        plain = tiny_model().train()
        plain.set_tpca_step(4)
        open_gates(plain)
        checkpointed = copy.deepcopy(plain)
        checkpointed.checkpoint_activations = True
        args = inputs()
        outputs = []
        for model in (plain, checkpointed):
            out, aux = model(**args)
            aux = aux["__tpca__"]
            loss = out.square().mean() + .03 * aux["ctc_loss"] + .01 * aux["path_loss"]
            assert torch.isfinite(loss) and aux["path_loss"] > 0
            loss.backward()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    assert torch.isfinite(param.grad).all(), name
            assert model.tpca_aligner.output_proj.weight.grad.abs().sum() > 0
            assert model.transformer_blocks[0].cross_attn.audio_cross_q.weight.grad.abs().sum() > 0
            outputs.append((out, loss))
        for left, right in zip(outputs[0], outputs[1]):
            torch.testing.assert_close(left, right)
        for (name, left), (_, right) in zip(plain.named_parameters(), checkpointed.named_parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, msg=name)

    def test_alignment_cache_clears_and_rejects_changed_inputs(self):
        model = tiny_model().eval()
        model.set_tpca_step(4)
        args = inputs()
        calls = []
        hook = model.tpca_aligner.register_forward_hook(lambda *_: calls.append(1))
        with torch.no_grad():
            model(**args, cache=True)
            model(**args, cache=True)
            assert len(calls) == 1
            args["video"].add_(.1)
            model(**args, cache=True)
            assert len(calls) == 2
            model.clear_cache()
            model(**args, cache=True)
            assert len(calls) == 3
        hook.remove()

    def test_empty_text_ca_backward_is_finite(self):
        ca = HunyuanDualTextCrossAttention(64, 4, 32)
        mask = torch.tensor([[True, True, True], [False, False, False]])
        outputs = ca(torch.randn(2, 8, 64), torch.randn(2, 2, 64),
                     torch.randn(2, 3, 32), text_mask=mask)
        sum(out.square().mean() for out in outputs).backward()
        for out in outputs:
            assert torch.isfinite(out).all() and out[1].abs().sum() == 0
        for parameter in ca.parameters():
            assert parameter.grad is None or torch.isfinite(parameter.grad).all()

    def test_inference_mode_and_prompt_occurrence_span(self):
        model = tiny_model().eval()
        model.set_tpca_step(4)
        with torch.inference_mode():
            args = inputs()
            args["text"] = torch.tensor([[1, 2, 1, 2], [2, 3, 2, 3]])
            args["text_mask"] = torch.ones(2, 4, dtype=torch.bool)
            starts = torch.tensor([2, 2])
            alignment = model._get_tpca_alignment(
                args["video"], args["text"], args["video_mask"], args["text_mask"],
                torch.tensor([1, 1]), starts, True,
            )
            assert alignment["prior"][..., :2].abs().sum() == 0
            out, _ = model(**args, cache=True, tpca_text_start=starts, tpca_video_start=torch.tensor([1, 1]))
            assert torch.isfinite(out).all()


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
