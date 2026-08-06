import unittest

from aligndit.script.eval.eval_semantic_vae_warmstart_dev import (
    deterministic_draw,
    make_batch_plan,
    select_records,
    stable_seed,
    summarize_rows,
)


class DeterministicProtocolTest(unittest.TestCase):
    def test_draw_is_keyed_and_does_not_use_global_rng(self):
        kwargs = {
            "utterance_key": "dev-clean/1/2/1-2-3",
            "frames": 101,
            "channels": 64,
            "eval_seed": 666,
            "repeat": 0,
            "mask_fraction_min": 0.7,
            "mask_fraction_max": 1.0,
        }
        first = deterministic_draw(**kwargs)
        second = deterministic_draw(**kwargs)
        self.assertEqual(first["mask_start"], second["mask_start"])
        self.assertEqual(first["mask_end"], second["mask_end"])
        self.assertEqual(first["diffusion_time"], second["diffusion_time"])
        self.assertTrue(first["x0"].equal(second["x0"]))
        self.assertGreaterEqual(first["masked_frames"], 70)
        self.assertLessEqual(first["masked_frames"], 101)
        self.assertNotEqual(
            stable_seed(666, "x0_noise", kwargs["utterance_key"]), stable_seed(667, "x0_noise", kwargs["utterance_key"])
        )

    def test_selection_is_balanced_and_manifest_ordered(self):
        records = [
            {"subset": subset, "utterance_key": f"{subset}/{index}"}
            for index in range(10)
            for subset in ("dev-clean", "dev-other")
        ]
        selected = select_records(records, eval_seed=666, limit_per_subset=3)
        self.assertEqual(sum(row["subset"] == "dev-clean" for row in selected), 3)
        self.assertEqual(sum(row["subset"] == "dev-other" for row in selected), 3)
        original_positions = [records.index(row) for row in selected]
        self.assertEqual(original_positions, sorted(original_positions))

    def test_batch_plan_limits_padded_cost(self):
        records = [{"latent_frames": frames} for frames in (100, 300, 100, 100, 400)]
        batches = make_batch_plan(records, padded_frame_budget=600, max_samples=3)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])
        for batch in batches:
            self.assertLessEqual(max(row["latent_frames"] for row in batch) * len(batch), 600)

    def test_batch_plan_rejects_one_oversized_utterance(self):
        with self.assertRaisesRegex(ValueError, "exceeding padded_frame_budget"):
            make_batch_plan([{"latent_frames": 601}], padded_frame_budget=600, max_samples=1)


class SummaryTest(unittest.TestCase):
    @staticmethod
    def row(key, subset, flow, cosine, count=64):
        return {
            "flow_element_count": count,
            "flow_mse": flow,
            "flow_squared_error_sum": flow * count,
            "hubert_cosine": cosine,
            "hubert_cosine_sum": cosine * 10,
            "hubert_frame_count": 10,
            "subset": subset,
            "utterance_key": key,
        }

    def test_micro_macro_and_repeat_aggregation(self):
        rows = [
            self.row("a", "dev-clean", 1.0, 0.4),
            self.row("a", "dev-clean", 3.0, 0.6),
            self.row("b", "dev-other", 5.0, 0.8, count=128),
        ]
        summary = summarize_rows(rows, bootstrap_samples=0, bootstrap_seed=1)
        self.assertAlmostEqual(summary["overall"]["flow_mse_macro"], 3.5)
        self.assertAlmostEqual(summary["overall"]["hubert_cosine_macro"], 0.65)
        self.assertAlmostEqual(summary["dev-clean"]["flow_mse_macro"], 2.0)
        self.assertEqual(summary["overall"]["utterances"], 2)


if __name__ == "__main__":
    unittest.main()
