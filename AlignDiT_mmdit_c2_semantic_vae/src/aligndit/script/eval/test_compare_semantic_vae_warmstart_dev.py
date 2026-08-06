import hashlib
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

from aligndit.script.eval.compare_semantic_vae_warmstart_dev import (
    compare_evaluations,
    render_markdown,
)


class WarmStartComparisonTest(unittest.TestCase):
    @staticmethod
    def _rows(flow_by_key, cosine_by_key):
        rows = []
        metadata = {
            "clean/a": ("dev-clean", "speaker-a", 100),
            "other/b": ("dev-other", "speaker-b", 120),
        }
        for repeat in range(2):
            for key in ("clean/a", "other/b"):
                subset, speaker, frames = metadata[key]
                masked_frames = 80 + repeat
                rows.append(
                    {
                        "diffusion_time": 0.25 + repeat * 0.1,
                        "flow_element_count": masked_frames * 64,
                        "flow_mse": flow_by_key[key][repeat],
                        "frames": frames,
                        "hubert_cosine": cosine_by_key[key][repeat],
                        "hubert_frame_count": frames,
                        "mask_end": 10 + masked_frames,
                        "mask_fraction_realized": masked_frames / frames,
                        "mask_fraction_sampled": 0.8 + repeat * 0.01,
                        "mask_start": 10,
                        "masked_frames": masked_frames,
                        "repeat": repeat,
                        "speaker_id": speaker,
                        "subset": subset,
                        "utterance_key": key,
                    }
                )
        return rows

    @staticmethod
    def _group_summary(rows, subset=None):
        selected = rows if subset is None else [row for row in rows if row["subset"] == subset]
        by_key = defaultdict(list)
        for row in selected:
            by_key[row["utterance_key"]].append(row)
        flow = [sum(row["flow_mse"] for row in draws) / len(draws) for draws in by_key.values()]
        cosine = [sum(row["hubert_cosine"] for row in draws) / len(draws) for draws in by_key.values()]
        return {
            "draws": len(selected),
            "flow_mse_macro": sum(flow) / len(flow),
            "hubert_cosine_macro": sum(cosine) / len(cosine),
            "utterances": len(by_key),
        }

    def _write_pair(self, directory, label, rows, *, eval_seed=666):
        ordered_keys = []
        for row in rows:
            if row["utterance_key"] not in ordered_keys:
                ordered_keys.append(row["utterance_key"])
        summary = {
            "checkpoint": {"sha256": f"sha-{label}", "stage": "s2c", "update": 70000, "weights": "ema"},
            "dataset": {
                "hubert_completion_sha256": "hubert",
                "latent_completion_sha256": "latent",
                "manifest_count": 5551,
                "manifest_sha256": "manifest",
                "normalization_sha256": "normalization",
                "selected_count": 2,
                "selected_counts": {"dev-clean": 1, "dev-other": 1},
                "selected_keys_sha256": hashlib.sha256("\n".join(ordered_keys).encode()).hexdigest(),
                "subset_counts": {"dev-clean": 2694, "dev-other": 2857},
            },
            "label": label,
            "protocol": {
                "condition_mode": "masked_audio",
                "eval_seed": eval_seed,
                "mask_fraction": [0.7, 1.0],
                "name": "test-protocol",
                "padded_frame_budget": 4000,
                "repeats": 2,
            },
            "results": {
                "overall": self._group_summary(rows),
                "dev-clean": self._group_summary(rows, "dev-clean"),
                "dev-other": self._group_summary(rows, "dev-other"),
            },
            "schema_version": 1,
        }
        summary_path = directory / f"{label}.summary.json"
        rows_path = directory / f"{label}.per_utterance.jsonl"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        rows_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return summary_path, rows_path

    def test_paired_deltas_ranks_and_markdown(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            baseline_rows = self._rows(
                {"clean/a": [2.0, 4.0], "other/b": [6.0, 8.0]},
                {"clean/a": [0.4, 0.6], "other/b": [0.2, 0.4]},
            )
            candidate_rows = self._rows(
                {"clean/a": [1.0, 3.0], "other/b": [4.0, 6.0]},
                {"clean/a": [0.5, 0.7], "other/b": [0.35, 0.55]},
            )
            baseline, _ = self._write_pair(directory, "baseline", baseline_rows)
            candidate, _ = self._write_pair(directory, "candidate", candidate_rows)
            comparison = compare_evaluations(
                [candidate, baseline], baseline_label="baseline", bootstrap_samples=200, bootstrap_seed=9
            )
            overall = {entry["label"]: entry for entry in comparison["groups"]["overall"]}
            self.assertAlmostEqual(overall["candidate"]["flow_delta_vs_baseline"]["estimate"], -1.5)
            self.assertAlmostEqual(overall["candidate"]["hubert_cosine_delta_vs_baseline"]["estimate"], 0.125)
            self.assertEqual(overall["candidate"]["flow_rank"], 1)
            self.assertEqual(overall["candidate"]["hubert_cosine_rank"], 1)
            flow_ci = overall["candidate"]["flow_delta_vs_baseline"]["ci95"]
            cosine_ci = overall["candidate"]["hubert_cosine_delta_vs_baseline"]["ci95"]
            self.assertLess(flow_ci[1], 0)
            self.assertGreater(cosine_ci[0], 0)
            markdown = render_markdown(comparison, sort_by="flow")
            self.assertIn("Flow deltas are candidate - baseline", markdown)
            self.assertIn("`candidate`", markdown)

    def test_protocol_and_paired_metadata_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            rows = self._rows(
                {"clean/a": [2.0, 4.0], "other/b": [6.0, 8.0]},
                {"clean/a": [0.4, 0.6], "other/b": [0.2, 0.4]},
            )
            baseline, _ = self._write_pair(directory, "baseline", rows)
            protocol_mismatch, _ = self._write_pair(directory, "protocol-mismatch", rows, eval_seed=667)
            with self.assertRaisesRegex(ValueError, "Protocol mismatch"):
                compare_evaluations(
                    [baseline, protocol_mismatch],
                    baseline_label="baseline",
                    bootstrap_samples=10,
                    bootstrap_seed=1,
                )

            altered_rows = [dict(row) for row in rows]
            altered_rows[0]["mask_start"] += 1
            metadata_mismatch, _ = self._write_pair(directory, "metadata-mismatch", altered_rows)
            with self.assertRaisesRegex(ValueError, "Paired draw metadata mismatch"):
                compare_evaluations(
                    [baseline, metadata_mismatch],
                    baseline_label="baseline",
                    bootstrap_samples=10,
                    bootstrap_seed=1,
                )


if __name__ == "__main__":
    unittest.main()
