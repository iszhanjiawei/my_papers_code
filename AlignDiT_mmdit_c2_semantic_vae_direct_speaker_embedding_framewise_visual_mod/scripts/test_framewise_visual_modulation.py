"""CPU behavior tests for the isolated framewise visual modulation experiment.

Run from this experiment root with ``PYTHONPATH=src python -m unittest
discover -s scripts -p test_framewise_visual_modulation.py -v``. No datasets,
teacher models, checkpoints or GPU are needed. Nonzero parent residual gates
and output weights simulate an already trained checkpoint, avoiding tests
which pass only because the scratch network predicts zero everywhere.
"""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.utils.checkpoint

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT


ARCH = {
    "dim": 32,
    "depth": 4,
    "heads": 4,
    "dim_head": 8,
    "dropout": 0.0,
    "ff_mult": 2,
    "mel_dim": 64,
    "text_num_embeds": 16,
    "text_dim": 16,
    "text_mask_padding": False,
    "qk_norm": "rms_norm",
    "conv_layers": 1,
    "pe_attn_head": 1,
    "attn_mask_enabled": True,
    "checkpoint_activations": False,
    "use_conformer": False,
    "layer_indices_ctc": [1, 2],
    "ctc_sampling_ratios": [1, 1],
    "n_mm_layers": 2,
    "n_text_layers": 2,
    "prompt_isolated_ca": False,
    "audio_video_ratio": 1,
    "video_dim": 16,
    "video_rope_scaled": False,
    "speaker_dim": 192,
    "speaker_condition_start_layer": 2,
}


def inputs():
    audio_mask = torch.arange(12)[None, :] < torch.tensor([12, 10])[:, None]
    text_mask = torch.arange(4)[None, :] < torch.tensor([4, 3])[:, None]
    generation = audio_mask.clone()
    generation[:, :3] = False
    return {
        "x": torch.randn(2, 12, 64),
        "cond": torch.randn(2, 12, 64),
        "text": torch.randint(0, 16, (2, 4)).masked_fill(~text_mask, -1),
        "video": torch.randn(2, 12, 16),
        "time": torch.tensor([0.2, 0.8]),
        "mask": audio_mask,
        "text_mask": text_mask,
        "video_mask": audio_mask.clone(),
        "complementary_mask": audio_mask & ~generation,
        "generation_mask": generation,
        "speaker_embedding": torch.randn(2, 192),
        "cache": False,
    }


def make_pair(*, mode="framewise", checkpoint=False):
    torch.manual_seed(101)
    parent = DiT_VT_MMDiT(**ARCH)
    with torch.no_grad():
        for block in parent.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        for block in parent.transformer_blocks[: parent.n_mm_layers]:
            block.v_attn_norm.linear.weight.normal_(std=0.03)
            block.v_attn_norm.linear.bias.normal_(std=0.03)
            block.cross_attn_ada.weight.normal_(std=0.03)
            block.cross_attn_ada.bias.normal_(std=0.03)
        parent.proj_out.weight.normal_(std=0.03)
        parent.norm_out.linear.weight.normal_(std=0.03)
        parent.speaker_proj.weight.normal_(std=0.03)
    model = DiT_VT_MMDiT(
        **{**ARCH, "checkpoint_activations": checkpoint},
        framewise_visual_modulation=True,
        visual_modulation_mode=mode,
        visual_modulation_hidden_dim=16,
    )
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    assert not unexpected
    assert len(missing) == 2 + 4 * ARCH["n_mm_layers"]
    assert all("visual_modulation" in name for name in missing)
    return parent, model


def activate_modulation(model):
    with torch.no_grad():
        for block in model.transformer_blocks[: model.n_mm_layers]:
            block.visual_modulation.output.weight.normal_(std=0.05)
            block.visual_modulation.output.bias.normal_(std=0.02)


def condition(model, batch):
    return model.get_visual_modulation_condition(
        batch["video"],
        batch["generation_mask"],
        audio_mask=batch["mask"],
        video_mask=batch["video_mask"],
        complementary_mask=batch["complementary_mask"],
    )


class FramewiseVisualModulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(37)

    def assert_nonzero_finite(self, tensor):
        self.assertIsNotNone(tensor)
        self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertGreater(torch.count_nonzero(tensor).item(), 0)

    def test_disabled_matches_unchanged_baseline_source(self):
        # The untouched sibling is read only, never edited or imported as the
        # active aligndit package. This catches shared-code regressions which
        # enabled-vs-disabled comparisons within the new file cannot detect.
        project = Path(__file__).resolve().parents[1]
        baseline_path = (
            project.parent
            / "AlignDiT_mmdit_c2_semantic_vae_direct_speaker_embedding"
            / "src/aligndit/model/backbone/dit_vt_mm.py"
        )
        spec = importlib.util.spec_from_file_location("unaltered_speaker_backbone", baseline_path)
        baseline_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline_module)
        torch.manual_seed(77)
        original_init = baseline_module.DiT_VT_MMDiT(**ARCH).state_dict()
        for enabled in (False, True):
            torch.manual_seed(77)
            migrated_init = DiT_VT_MMDiT(**ARCH, framewise_visual_modulation=enabled).state_dict()
            if not enabled:
                self.assertEqual(set(migrated_init), set(original_init))
            for name, value in original_init.items():
                torch.testing.assert_close(migrated_init[name], value, rtol=0, atol=0, msg=name)
        disabled, _ = make_pair()
        baseline = baseline_module.DiT_VT_MMDiT(**ARCH)
        baseline.load_state_dict(disabled.state_dict(), strict=True)
        baseline.eval()
        disabled.eval()
        batch = inputs()
        for flags in ({}, {"cfg_infer": True}, {"cfg_infer": True, "drop_video": True},
                      {"cfg_infer": True, "drop_text": True}):
            with self.subTest(flags=flags), torch.inference_mode():
                expected, expected_ctc = baseline(**batch, **flags)
                actual, actual_ctc = disabled(**batch, **flags)
                self.assertGreater(expected.square().mean().sqrt().item(), 0.01)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for layer in expected_ctc:
                    torch.testing.assert_close(
                        actual_ctc[layer]["z_tilde"], expected_ctc[layer]["z_tilde"], rtol=0, atol=0,
                    )

    def test_zero_initialized_modulation_preserves_parent_and_cfg(self):
        parent, model = make_pair()
        parent.eval()
        model.eval()
        batch = inputs()
        for flags in ({}, {"cfg_infer": True}, {"cfg_infer": True, "drop_video": True},
                      {"cfg_infer": True, "drop_text": True}):
            with self.subTest(flags=flags), torch.inference_mode():
                expected, old_ctc = parent(**batch, **flags)
                actual, new_ctc = model(**batch, **flags)
                self.assert_nonzero_finite(expected)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                for layer in old_ctc:
                    torch.testing.assert_close(new_ctc[layer]["z_tilde"], old_ctc[layer]["z_tilde"], rtol=0, atol=0)
                    torch.testing.assert_close(new_ctc[layer]["z_lens"], old_ctc[layer]["z_lens"], rtol=0, atol=0)

    def test_condition_masks_exclude_prompt_padding_and_complement(self):
        for mode in ("framewise", "pooled"):
            with self.subTest(mode=mode):
                _, model = make_pair(mode=mode)
                activate_modulation(model)
                batch = inputs()
                batch["complementary_mask"][0, 6] = True
                features, valid = condition(model, batch)
                expected_valid = batch["generation_mask"] & batch["mask"] & batch["video_mask"]
                expected_valid &= ~batch["complementary_mask"]
                self.assertTrue(torch.equal(valid, expected_valid))
                self.assertEqual(torch.count_nonzero(features[~valid]).item(), 0)
                contaminated = {**batch, "video": batch["video"].clone()}
                # Invalid features can even contain NaNs; they must not leak
                # into the new visual path or pooled statistics.
                contaminated["video"][~valid] = float("nan")
                cleaned_features, cleaned_valid = condition(model, contaminated)
                torch.testing.assert_close(cleaned_features, features, rtol=0, atol=0)
                self.assertTrue(torch.equal(cleaned_valid, valid))
                block = model.transformer_blocks[0].visual_modulation
                delta = block(features, model.time_embed(batch["time"]), valid)
                self.assert_nonzero_finite(delta[valid])
                self.assertEqual(torch.count_nonzero(delta[~valid]).item(), 0)
                empty = {**batch, "generation_mask": torch.zeros_like(valid)}
                empty_features, empty_valid = condition(model, empty)
                empty_delta = block(empty_features, model.time_embed(batch["time"]), empty_valid)
                self.assertTrue(torch.isfinite(empty_delta).all().item())
                self.assertEqual(torch.count_nonzero(empty_delta).item(), 0)

    def test_framewise_preserves_correspondence_and_pooled_is_order_invariant(self):
        _, framewise = make_pair()
        pooled = copy.deepcopy(framewise)
        pooled.visual_modulation_mode = "pooled"
        batch = inputs()
        swapped = {**batch, "video": batch["video"].clone()}
        swapped["video"][:, [4, 7]] = swapped["video"][:, [7, 4]]
        f0, _ = condition(framewise, batch)
        f1, _ = condition(framewise, swapped)
        p0, valid = condition(pooled, batch)
        p1, _ = condition(pooled, swapped)
        self.assert_nonzero_finite(f0[:, 4] - f1[:, 4])
        torch.testing.assert_close(f0[:, 4], f1[:, 7], rtol=0, atol=0)
        torch.testing.assert_close(f0[:, [3, 5, 6]], f1[:, [3, 5, 6]], rtol=0, atol=0)
        torch.testing.assert_close(p0, p1, rtol=1e-6, atol=1e-7)
        for index in range(2):
            rows = p0[index, valid[index]]
            torch.testing.assert_close(rows, rows[:1].expand_as(rows), rtol=0, atol=0)

    def test_nonzero_modulation_responds_to_video_and_time(self):
        parent, model = make_pair()
        activate_modulation(model)
        parent.eval()
        model.eval()
        batch = inputs()
        features, valid = condition(model, batch)
        block = model.transformer_blocks[0].visual_modulation
        delta = block(features, model.time_embed(batch["time"]), valid)
        video_delta = block(features.flip(1), model.time_embed(batch["time"]), valid)
        time_delta = block(features, model.time_embed(1 - batch["time"]), valid)
        self.assert_nonzero_finite(delta - video_delta)
        self.assert_nonzero_finite(delta - time_delta)
        with torch.inference_mode():
            old_output, _ = parent(**batch)
            new_output, _ = model(**batch)
        self.assert_nonzero_finite(new_output - old_output)

    def test_short_video_is_padded_without_resampling_and_long_video_rejected(self):
        _, model = make_pair()
        batch = inputs()
        features, valid = condition(model, batch)
        short = dict(batch)
        for key in ("video", "video_mask", "complementary_mask"):
            short[key] = batch[key][:, :8]
        shorter_features, shorter_valid = condition(model, short)
        torch.testing.assert_close(shorter_features[:, :8], features[:, :8], rtol=0, atol=0)
        self.assertTrue(torch.equal(shorter_valid[:, :8], valid[:, :8]))
        self.assertEqual(torch.count_nonzero(shorter_features[:, 8:]).item(), 0)
        self.assertFalse(shorter_valid[:, 8:].any().item())
        with self.assertRaises(ValueError):
            condition(model, {**batch, "video": torch.randn(2, 13, 16)})
        with self.assertRaises(ValueError):
            DiT_VT_MMDiT(**{**ARCH, "audio_video_ratio": 4}, framewise_visual_modulation=True)

    def test_packed_cfg_equals_separate_branches_for_batch_two(self):
        _, model = make_pair()
        activate_modulation(model)
        model.eval()
        batch = inputs()
        null_flags = {"drop_audio_cond": True, "drop_text": True, "drop_video": True}
        for flags, singles in (
            ({}, ({}, {"drop_video": True}, null_flags)),
            ({"drop_video": True}, ({"drop_video": True}, null_flags)),
            ({"drop_text": True}, ({"drop_text": True}, null_flags)),
        ):
            with self.subTest(flags=flags), torch.inference_mode():
                packed, _ = model(**batch, cfg_infer=True, **flags)
                separate = torch.cat([model(**batch, **branch)[0] for branch in singles])
                torch.testing.assert_close(packed, separate, rtol=2e-5, atol=2e-6)
        altered = {**batch, "video": torch.randn_like(batch["video"]) * 7}
        with torch.inference_mode():
            first, _ = model(**batch, drop_video=True)
            second, _ = model(**altered, drop_video=True)
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_flow_gradient_reaches_zero_head_then_visual_and_time_inputs(self):
        for checkpoint in (False, True):
            with self.subTest(checkpoint=checkpoint):
                _, model = make_pair(checkpoint=checkpoint)
                model.train()
                batch = inputs()
                target = torch.randn_like(batch["x"])
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    output, _ = model(**batch)
                    (output[batch["generation_mask"]] - target[batch["generation_mask"]]).square().mean().backward()
                    for block in model.transformer_blocks[: model.n_mm_layers]:
                        self.assert_nonzero_finite(block.visual_modulation.output.weight.grad)
                        if step == 1:
                            self.assert_nonzero_finite(block.visual_modulation.time_proj.weight.grad)
                    if step == 1:
                        self.assert_nonzero_finite(model.visual_modulation_input[1].weight.grad)
                    optimizer.step()
                # A dropped video still executes the modules so DDP observes
                # their parameters, but gradients are exactly zero.
                optimizer.zero_grad(set_to_none=True)
                output, _ = model(**batch, drop_video=True)
                output.square().mean().backward()
                for name, parameter in model.named_parameters():
                    if "visual_modulation" in name:
                        self.assertIsNotNone(parameter.grad, name)
                        self.assertEqual(torch.count_nonzero(parameter.grad).item(), 0, name)

    def test_checkpointing_preserves_outputs_and_modulation_gradients(self):
        _, regular = make_pair()
        activate_modulation(regular)
        checkpointed = copy.deepcopy(regular)
        checkpointed.checkpoint_activations = True
        regular.train()
        checkpointed.train()
        batch = inputs()
        expected, _ = regular(**batch)
        actual, _ = checkpointed(**batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected.square().mean().backward()
        actual.square().mean().backward()
        other = dict(checkpointed.named_parameters())
        for name, parameter in regular.named_parameters():
            if "visual_modulation" in name:
                self.assert_nonzero_finite(parameter.grad)
                torch.testing.assert_close(other[name].grad, parameter.grad, rtol=2e-5, atol=1e-8)

    def test_cfm_training_retains_flow_ctc_and_modulation_gradient(self):
        _, transformer = make_pair()
        model = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1,
            ctc_lambda=0.03, audio_drop_prob=0.0, cond_drop_prob=0.0,
            text_drop_prob=0.0, video_drop_prob=0.0,
        )
        batch = inputs()
        with patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, prediction = model(
                inp=batch["x"], text=batch["text"], video=batch["video"],
                lens=torch.tensor([12, 10]), text_lens=torch.tensor([4, 3]),
                video_lens=torch.tensor([12, 10]), speaker_embedding=batch["speaker_embedding"],
            )
        self.assertEqual(set(components), {"diff_loss", "ctc_loss"})
        self.assertAlmostEqual(loss.item(), components["diff_loss"] + 0.03 * components["ctc_loss"], places=6)
        self.assertTrue(torch.isfinite(prediction).all().item())
        loss.backward()
        self.assert_nonzero_finite(transformer.transformer_blocks[0].visual_modulation.output.weight.grad)

    def test_cfm_sampling_cfg_branches_and_short_video(self):
        _, transformer = make_pair()
        activate_modulation(transformer)
        model = CFM_VT(
            transformer=transformer, num_channels=64, audio_video_ratio=1,
            ctc_lambda=0.03, odeint_kwargs={"method": "euler"},
        )
        batch = inputs()
        common = {
            "cond": batch["cond"][:, :3], "text": batch["text"], "duration": torch.tensor([12, 10]),
            "video": batch["video"][:, :8], "lens": torch.tensor([3, 3]),
            "speaker_embedding": batch["speaker_embedding"], "steps": 2, "use_epss": False, "seed": 0,
        }
        for ignore in (None, "video", "text"):
            with self.subTest(ignore=ignore):
                sampled, trajectory = model.sample(
                    **common, cfg_strength=1.0, cfg_strength_v=1.0, ignore_modality=ignore,
                )
                self.assertEqual(sampled.shape, (2, 12, 64))
                self.assertTrue(torch.isfinite(trajectory).all().item())
                torch.testing.assert_close(sampled[:, :3], common["cond"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
