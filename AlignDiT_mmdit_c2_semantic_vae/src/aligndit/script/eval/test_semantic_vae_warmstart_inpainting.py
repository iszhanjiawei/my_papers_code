import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from aligndit.script.eval.decode_semantic_vae_warmstart_inpainting import (
    decode,
    expected_inpainting_span,
    inverse_normalize,
    latent_error_metrics,
    si_sdr,
    summarize,
    validate_exact_length,
)
from aligndit.script.eval.generate_semantic_vae_warmstart_inpainting import (
    atomic_save_npy,
    build_keep_mask,
    fixed_inpainting_span,
    select_inpainting_records,
    stable_seed,
    validate_generation_length,
)


class SelectionAndMaskTest(unittest.TestCase):
    @staticmethod
    def records():
        records = []
        for subset in ("dev-clean", "dev-other"):
            for index in range(8):
                records.append(
                    {
                        "duration_seconds": 4.0 + index / 2,
                        "speaker_id": str(index // 2),
                        "subset": subset,
                        "utterance_key": f"{subset}/{index // 2}/1/{index}",
                    }
                )
            records.append(
                {
                    "duration_seconds": 2.0,
                    "speaker_id": "short",
                    "subset": subset,
                    "utterance_key": f"{subset}/short/1/0",
                }
            )
        return records

    def test_selection_is_balanced_unique_speaker_and_order_independent(self):
        records = self.records()
        kwargs = {
            "eval_seed": 666,
            "limit_per_subset": 3,
            "min_duration": 4.0,
            "max_duration": 10.0,
        }
        selected = select_inpainting_records(records, **kwargs)
        reversed_selected = select_inpainting_records(list(reversed(records)), **kwargs)
        self.assertEqual(
            [record["utterance_key"] for record in selected],
            [record["utterance_key"] for record in reversed_selected],
        )
        self.assertEqual(len(selected), 6)
        for subset in ("dev-clean", "dev-other"):
            subset_rows = [record for record in selected if record["subset"] == subset]
            self.assertEqual(len(subset_rows), 3)
            self.assertEqual(len({record["speaker_id"] for record in subset_rows}), 3)
            self.assertTrue(all(4.0 <= record["duration_seconds"] <= 10.0 for record in subset_rows))

    def test_fixed_span_is_deterministic_and_realizes_floor_fraction(self):
        kwargs = {"utterance_key": "dev-clean/1/2/1-2-3", "frames": 101, "mask_fraction": 0.7, "eval_seed": 666}
        first = fixed_inpainting_span(**kwargs)
        second = fixed_inpainting_span(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first[1] - first[0], 70)
        self.assertGreaterEqual(first[0], 0)
        self.assertLessEqual(first[1], 101)
        self.assertEqual(stable_seed(666, "inpainting_ode_noise", kwargs["utterance_key"]), 7979176747244629837)
        self.assertEqual(
            first,
            expected_inpainting_span(
                utterance_key=kwargs["utterance_key"],
                frames=kwargs["frames"],
                mask_fraction=kwargs["mask_fraction"],
                eval_seed=kwargs["eval_seed"],
            ),
        )

    def test_keep_mask_matches_sampler_semantics(self):
        keep = build_keep_mask(frames=7, mask_start=2, mask_end=5, device=torch.device("cpu"))
        self.assertEqual(keep.tolist(), [[True, True, False, False, False, True, True]])

    def test_generation_length_enforces_40_hz_ceil_contract(self):
        validate_generation_length(frames=3, original_samples=801, padded_samples=1200)
        with self.assertRaisesRegex(ValueError, "ceil"):
            validate_generation_length(frames=3, original_samples=800, padded_samples=1200)

    def test_atomic_npy_refuses_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested/value.npy"
            value = np.arange(12, dtype=np.float32).reshape(3, 4)
            atomic_save_npy(path, value)
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), value)
            with self.assertRaises(FileExistsError):
                atomic_save_npy(path, value)


class DecodeMathTest(unittest.TestCase):
    def test_decode_uses_the_standalone_decoder_dataset_validator(self):
        generation = {
            "dataset": {"cache_root": "cache", "manifest": "manifest", "normalization": "normalization"},
            "label": "test",
            "protocol": {"eval_seed": 666},
        }
        with tempfile.TemporaryDirectory() as directory:
            generation_dir = Path(directory) / "generation"
            generation_dir.mkdir()
            args = type(
                "Args",
                (),
                {
                    "device": "cpu",
                    "generation_dir": generation_dir,
                    "output_dir": Path(directory) / "output",
                    "label": "test",
                },
            )()
            with (
                patch.dict(sys.modules, {"pystoi": types.SimpleNamespace(stoi=lambda *_args, **_kwargs: 0.0)}),
                patch("aligndit.script.eval.decode_semantic_vae_warmstart_inpainting.configure_runtime"),
                patch(
                    "aligndit.script.eval.decode_semantic_vae_warmstart_inpainting.load_generation",
                    return_value=([{"utterance_key": "unused"}], generation),
                ),
                patch(
                    "aligndit.script.eval.decode_semantic_vae_warmstart_inpainting.validate_decoder_dataset",
                    side_effect=RuntimeError("validator-called"),
                ) as validator,
                self.assertRaisesRegex(RuntimeError, "validator-called"),
            ):
                decode(args)
        validator.assert_called_once_with(generation["dataset"], selected_keys={"unused"})

    def test_inverse_normalization_and_masked_latent_metrics(self):
        target = np.ones((4, 64), dtype=np.float32)
        generated = target.copy()
        generated[1:3] += 2.0
        mean = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
        std = np.linspace(0.5, 1.5, 64, dtype=np.float32)
        raw = inverse_normalize(generated, mean, std)
        np.testing.assert_allclose(raw, generated * std + mean)
        metrics = latent_error_metrics(generated, target, 1, 3)
        self.assertAlmostEqual(metrics["latent_mse"], 4.0)
        self.assertAlmostEqual(metrics["latent_mae"], 2.0)
        self.assertTrue(-1.0 <= metrics["latent_cosine"] <= 1.0)

    def test_exact_length_uses_ceil_hop_contract(self):
        self.assertEqual(validate_exact_length(3, 801, 1200), 1200)
        self.assertEqual(validate_exact_length(3, 1200, 1200), 1200)
        with self.assertRaisesRegex(ValueError, "padded_samples"):
            validate_exact_length(3, 801, 1199)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            validate_exact_length(3, 800, 1200)

    def test_si_sdr_is_scale_invariant(self):
        reference = np.sin(np.linspace(0, 12, 1600, dtype=np.float64)).astype(np.float32)
        self.assertGreater(si_sdr(reference, reference * 2.0), 80.0)

    def test_summary_separates_subsets(self):
        rows = []
        for index, subset in enumerate(("dev-clean", "dev-other")):
            rows.append(
                {
                    "latent_cosine": 0.5 + index / 10,
                    "latent_mae": 0.2 + index / 10,
                    "latent_absolute_error_sum": (0.2 + index / 10) * 64,
                    "latent_cosine_sum": (0.5 + index / 10),
                    "latent_element_count": 64,
                    "latent_frame_count": 1,
                    "latent_mse": 0.1 + index / 10,
                    "latent_squared_error_sum": (0.1 + index / 10) * 64,
                    "masked_audio_samples": 400,
                    "masked_si_sdr_db": 2.0 + index,
                    "masked_stoi": 0.7 + index / 10,
                    "subset": subset,
                }
            )
        summary = summarize(rows)
        self.assertEqual(summary["overall"]["utterances"], 2)
        self.assertEqual(summary["dev-clean"]["utterances"], 1)
        self.assertAlmostEqual(summary["overall"]["latent_mse_mean"], 0.15)


if __name__ == "__main__":
    unittest.main()
