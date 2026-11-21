"""Structural tests for transcript routing, conditioning dropout and target-only CTC."""

import unittest

import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, gather_target_frames


def tiny_backbone(*, framewise_reference=False, simulate_trained=True):
    model = DiT_VT_MMDiT(
        dim=32, depth=3, heads=2, dim_head=16, ff_mult=2, dropout=0.0,
        mel_dim=8, text_num_embeds=32, text_dim=16, conv_layers=1,
        use_conformer=False, layer_indices_ctc=[1], ctc_sampling_ratios=(1,),
        n_mm_layers=1, n_text_layers=2, prompt_isolated_ca=False,
        audio_video_ratio=1, video_dim=12, speaker_dim=4,
        role_conditioned_text=True, role_framewise_reference=framewise_reference, attn_mask_enabled=True,
    )
    if not simulate_trained:
        return model
    # Emulate a trained checkpoint: inherited residual gates and proj_out are
    # zero in an untrained toy DiT, making every final prediction identical.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if (param.numel() > 1 and torch.count_nonzero(param) == 0
                    and not name.endswith(".bias")):
                param.normal_(std=0.1)
    return model.eval()


def mixed_length_inputs():
    reference = torch.tensor([[4, 2, 3, -1], [1, 2, 3, 4]])
    target = torch.tensor([[1, 2, 3], [4, 5, -1]])
    valid = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]]).bool()
    generate = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 0, 0]]).bool()
    return dict(
        x=torch.randn(2, 8, 8), cond=torch.randn(2, 8, 8),
        text=target, reference_text=reference, video=torch.randn(2, 8, 12),
        time=torch.tensor([0.5, 0.9]), mask=valid, text_mask=target >= 0,
        video_mask=valid, complementary_mask=valid & ~generate, generation_mask=generate,
        speaker_embedding=torch.randn(2, 4),
    )


class RoleConditioningTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = tiny_backbone()
        self.kw = mixed_length_inputs()

    def test_reference_affects_only_pooled_style_and_conditional_prediction(self):
        model = self.model
        observed_keys, stem_inputs, proj_inputs = [], [], []
        hooks = [
            model.transformer_blocks[0].cross_attn.register_forward_pre_hook(
                lambda _, args: observed_keys.append(args[1].clone())
            ),
            model.input_embed.register_forward_pre_hook(
                lambda _, args: stem_inputs.append((args[0].clone(), args[1].clone()))
            ),
            model.projectors_ctc[0].register_forward_pre_hook(
                lambda _, args: proj_inputs.append((args[0].clone(), args[1].clone()))
            ),
        ]
        try:
            with torch.no_grad():
                baseline, ctc = model(**self.kw)
                changed = dict(self.kw, reference_text=torch.tensor([[5, 5, 6, 7, 8], [8, 7, -1, -1, -1]]))
                swapped, _ = model(**changed)
                packed, _ = model(**self.kw, cfg_infer=True, cache=True)
                model.clear_cache()
                packed_changed, _ = model(**changed, cfg_infer=True, cache=True)
                model.clear_cache()
                null, _ = model(**changed, drop_audio_cond=True, drop_text=True, drop_video=True)
        finally:
            for hook in hooks:
                hook.remove()

        torch.testing.assert_close(observed_keys[0], observed_keys[1], rtol=0, atol=0)
        self.assertFalse(torch.allclose(baseline, swapped, atol=1e-7))
        torch.testing.assert_close(packed[:2], baseline, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(packed_changed[:2], swapped, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(packed[-2:], packed_changed[-2:], atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(packed_changed[-2:], null, atol=1e-5, rtol=1e-5)
        stem_x, stem_cond = stem_inputs[0]
        self.assertEqual(stem_x[~self.kw["generation_mask"]].count_nonzero().item(), 0)
        self.assertEqual(stem_cond.count_nonzero().item(), 0)
        self.assertTrue(all(block.prompt_isolated_ca for block in model.transformer_blocks[:2]))
        self.assertEqual(proj_inputs[0][0].shape, (2, 6, 32))
        self.assertEqual(proj_inputs[0][1].tolist(), [6, 3])
        self.assertEqual(ctc[1]["z_lens"].tolist(), [6, 3])

    def test_framewise_reference_masks_actual_projection_input_in_cfg(self):
        model = tiny_backbone(framewise_reference=True)
        kw = self.kw
        generate = kw["generation_mask"]
        prompt = kw["mask"] & ~generate
        expected_x = kw["x"].masked_fill(~generate[..., None], 0)
        expected_cond = kw["cond"].masked_fill(~prompt[..., None], 0)
        observed = []
        hook = model.input_embed.proj.register_forward_pre_hook(
            lambda _, args: observed.append(args[0].detach().clone())
        )
        try:
            with torch.no_grad():
                model(**kw, cfg_infer=True)
                model(**kw, drop_audio_cond=True)
        finally:
            hook.remove()

        # Full and TTS branches see the prompt. CFG null and audio dropout see zeros.
        self.assertEqual(len(observed), 4)
        for idx, projected in enumerate(observed):
            x, cond = projected.chunk(2, dim=-1)
            torch.testing.assert_close(x, expected_x, atol=0, rtol=0)
            expected = expected_cond if idx < 2 else torch.zeros_like(expected_cond)
            torch.testing.assert_close(cond, expected, atol=0, rtol=0)

    def test_scratch_target_loss_trains_framewise_global_and_role_parameters(self):
        model = tiny_backbone(framewise_reference=True, simulate_trained=False)
        kw = self.kw
        channels = kw["cond"].shape[-1]
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        target_velocity = torch.randn_like(kw["x"])[kw["generation_mask"]]
        # The scratch output head starts at zero. Gradients reach the inner
        # projections after the head and the residual gates take their first updates.
        for _ in range(5):
            optimizer.zero_grad(set_to_none=True)
            out, _ = model(**kw)
            (out[kw["generation_mask"]] - target_velocity).square().mean().backward()
            optimizer.step()
        gradients = [
            model.input_embed.proj.weight.grad[:, channels:],
            model.speaker_proj.weight.grad,
            model.role_conditioner.reference_role.grad,
            model.role_conditioner.target_role.grad,
        ]
        for grad in gradients:
            self.assertTrue(grad is not None and grad.isfinite().all() and grad.norm() > 0)

    def test_target_gather_keeps_order_and_zeroes_padding(self):
        x = torch.arange(16).reshape(2, 8, 1).float().requires_grad_()
        selected, lens = gather_target_frames(x, self.kw["generation_mask"])
        self.assertEqual(selected.squeeze(-1).tolist(), [[2, 3, 4, 5, 6, 7], [11, 12, 13, 0, 0, 0]])
        self.assertEqual(lens.tolist(), [6, 3])
        selected.sum().backward()
        torch.testing.assert_close(x.grad.squeeze(-1), self.kw["generation_mask"].float())

    def test_text_role_and_style_gradients_on_regular_and_dropout_steps(self):
        model = self.model.train()
        for drop in (False, True):
            with self.subTest(drop=drop):
                model.zero_grad(set_to_none=True)
                out, _ = model(**self.kw, drop_text=drop, drop_audio_cond=drop, drop_video=drop)
                out[self.kw["generation_mask"]].square().mean().backward()
                for name, param in model.role_conditioner.named_parameters():
                    self.assertIsNotNone(param.grad, name)
                    self.assertTrue(torch.isfinite(param.grad).all(), name)
                    self.assertEqual(param.grad.abs().sum().item() > 0, not drop, name)


if __name__ == "__main__":
    unittest.main()
