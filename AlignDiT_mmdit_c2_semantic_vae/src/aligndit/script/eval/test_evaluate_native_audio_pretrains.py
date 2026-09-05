"""CPU tests for paired units, immutable manifests and physical waveform crops."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from aligndit.script.eval.evaluate_native_audio_pretrains import (
    BRANCHES,
    CHECKPOINTS,
    SEEDS,
    load_run,
    sha256_file,
    si_sdr,
    summarize_paired,
)


class PairedStatisticsTest(unittest.TestCase):
    def test_silent_prediction_is_not_reported_as_zero_db(self):
        reference = np.sin(np.arange(1000) * 0.1)
        self.assertEqual(si_sdr(reference, np.zeros_like(reference)), -120.0)
        self.assertGreater(si_sdr(reference, reference), 100.0)

    def rows(self):
        rows = []
        for branch in BRANCHES:
            for index in range(4):
                for seed in SEEDS:
                    rows.append(
                        {
                            "utterance_key": str(index),
                            "subset": "dev-clean" if index < 2 else "dev-other",
                            "speaker_id": str(index),
                            "branch": branch,
                            "sampling_seed": seed,
                            "score": float(index + seed - 666) + (0.125 if branch == "svae" else 0),
                            "context": None if index == 0 else float(index),
                        }
                    )
        return rows

    def test_repetitions_are_averaged_before_paired_utterance_bootstrap(self):
        result = summarize_paired(self.rows(), generated=True, bootstrap_samples=200, bootstrap_seed=9)
        self.assertEqual(result["overall"]["utterances"], 4)
        score = result["overall"]["metrics"]["score"]
        self.assertEqual(score["paired_utterances"], 4)
        self.assertEqual(score["mel_mean"], 2.5)
        self.assertEqual(score["svae_minus_mel"], 0.125)
        self.assertEqual(score["delta_ci95"], [0.125, 0.125])
        self.assertEqual(result["overall"]["metrics"]["context"]["paired_utterances"], 3)
        self.assertEqual(result["dev-clean"]["utterances"], 2)

    def test_missing_and_duplicate_draws_are_rejected(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "Incomplete seed"):
            summarize_paired(rows[:-1], generated=True, bootstrap_samples=100, bootstrap_seed=9)
        with self.assertRaisesRegex(ValueError, "Duplicate or missing"):
            summarize_paired(rows + [rows[0]], generated=True, bootstrap_samples=100, bootstrap_seed=9)

    def test_nonfinite_partial_or_mismatched_results_are_rejected(self):
        for value in (None, float("nan")):
            rows = self.rows()
            rows[0]["score"] = value
            with self.assertRaisesRegex(ValueError, "Partial/nonfinite"):
                summarize_paired(rows, generated=True, bootstrap_samples=100, bootstrap_seed=9)
        rows = self.rows()
        rows[-1]["subset"] = "dev-clean"
        with self.assertRaisesRegex(ValueError, "Inconsistent metric identity"):
            summarize_paired(rows, generated=True, bootstrap_samples=100, bootstrap_seed=9)


class BoundWaveformsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rows = []
        for index in range(50):
            wave = np.sin(np.arange(3200, dtype=np.float32) * 0.07) * 0.3
            full = self.wave(f"common/full/{index}.wav", wave)
            observed = wave.copy()
            observed[800:2400] = 0
            row = {
                "utterance_key": str(index),
                "subset": "dev-clean" if index < 25 else "dev-other",
                "speaker_id": str(index),
                "original_num_samples": 3200,
                "mask_start_sample": 800,
                "mask_end_sample": 2400,
                "source_audio_path": str(self.root / full["path"]),
                "source_audio_sha256": full["sha256"],
                "reference_full": full,
                "reference_input": self.wave(f"common/input/{index}.wav", observed),
                "reference_masked": self.wave(f"common/masked/{index}.wav", wave[800:2400]),
                "reference_context": self.wave(
                    f"common/context/{index}.wav", np.concatenate((wave[:800], wave[2400:]))
                ),
            }
            self.rows.append(row)
        common_manifest = self.manifest("common/manifest.jsonl", self.rows)
        self.common = {
            "schema_version": 1,
            "checkpoints": CHECKPOINTS,
            "manifest": common_manifest,
            "count": 50,
            "protocol": {
                "sampling_seeds": list(SEEDS),
                "name": "librispeech-native-mel500k-svae70k-waveform-masked-inpainting-v2",
                "conditioning": "same source waveform zeroed inside physical mask BEFORE native encoding; no text/video/HuBERT",
                "context_latent": "fixed keyed posterior sample of zero-masked waveform; clean cached latents are oracle only",
            },
        }
        self.json("common/complete.json", self.common)
        self.branch_rows = {}
        self.branch_completes = {}
        for branch in BRANCHES:
            rows = []
            for shared in self.rows:
                for seed in SEEDS:
                    row = copy.deepcopy(shared)
                    key = shared["utterance_key"]
                    full, _ = sf.read(self.root / shared["reference_full"]["path"], dtype="float32")
                    row.update(
                        branch=branch,
                        sampling_seed=seed,
                        oracle_full=shared["reference_full"],
                        oracle_masked=shared["reference_masked"],
                        generated_full=self.wave(f"{branch}/full/{key}/{seed}.wav", full),
                        generated_masked=self.wave(f"{branch}/masked/{key}/{seed}.wav", full[800:2400]),
                    )
                    rows.append(row)
            self.branch_rows[branch] = rows
            manifest = self.manifest(f"{branch}/generation_manifest.jsonl", rows)
            complete = {
                "branch": branch,
                "generation_manifest": manifest,
                "count": 150,
                "common_manifest_sha256": common_manifest["sha256"],
                "common_complete_sha256": sha256_file(self.root / "common/complete.json"),
                "protocol": self.common["protocol"],
                "waveform_complete": True,
                "canary_limit": None,
                "checkpoint": {"sha256": CHECKPOINTS[branch]},
            }
            self.branch_completes[branch] = complete
            self.json(f"{branch}/generation_complete.json", complete)

    def info(self, relative):
        path = self.root / relative
        return {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

    def wave(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, value, 16000, subtype="FLOAT")
        return self.info(relative)

    def manifest(self, relative, rows):
        (self.root / relative).write_text("".join(json.dumps(row) + "\n" for row in rows))
        return self.info(relative)

    def json(self, relative, value):
        (self.root / relative).write_text(json.dumps(value))

    def republish_mel(self):
        self.branch_completes["mel"]["generation_manifest"] = self.manifest(
            "mel/generation_manifest.jsonl", self.branch_rows["mel"]
        )
        self.json("mel/generation_complete.json", self.branch_completes["mel"])

    def test_complete_pair_is_accepted(self):
        common, branches, _, _ = load_run(self.root)
        self.assertEqual(len(common), 50)
        self.assertEqual(len(branches["mel"]), 150)

    def test_canary_uses_explicit_matching_marker_and_only_selected_utterances(self):
        for branch in BRANCHES:
            folder = f"{branch}_canary2"
            (self.root / folder).mkdir()
            complete = copy.deepcopy(self.branch_completes[branch])
            complete.update(canary_limit=2, count=6)
            complete["generation_manifest"] = self.manifest(
                f"{folder}/generation_manifest.jsonl", self.branch_rows[branch][:6]
            )
            self.json(f"{folder}/generation_complete.json", complete)
        common, branches, provenance, _ = load_run(self.root, canary_limit=2)
        self.assertEqual(len(common), 2)
        self.assertEqual(len(branches["mel"]), 6)
        self.assertEqual(provenance["canary_limit"], 2)

    def test_unmasked_encoder_input_is_rejected(self):
        row = self.rows[0]
        row["reference_input"] = row["reference_full"]
        self.common["manifest"] = self.manifest("common/manifest.jsonl", self.rows)
        self.json("common/complete.json", self.common)
        with self.assertRaisesRegex(ValueError, "Encoder input must replace"):
            load_run(self.root)

    def test_changed_wave_is_rejected_even_with_valid_manifest(self):
        row = self.branch_rows["mel"][0]
        self.wave(row["generated_masked"]["path"], np.zeros(1600, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Missing or changed waveform"):
            load_run(self.root)

    def test_authenticated_but_wrong_crop_is_rejected(self):
        row = self.branch_rows["mel"][0]
        row["generated_masked"] = self.wave(row["generated_masked"]["path"], np.zeros(1600, dtype=np.float32))
        self.republish_mel()
        with self.assertRaisesRegex(ValueError, "Wrong generated missing-region crop"):
            load_run(self.root)

    def test_duplicate_seed_and_mixed_physical_span_are_rejected(self):
        original = copy.deepcopy(self.branch_rows["mel"][0])
        self.branch_rows["mel"][0]["sampling_seed"] = SEEDS[1]
        self.republish_mel()
        with self.assertRaisesRegex(ValueError, "exactly 50 utterances"):
            load_run(self.root)
        self.branch_rows["mel"][0] = original
        self.branch_rows["mel"][0]["mask_end_sample"] += 800
        self.republish_mel()
        with self.assertRaisesRegex(ValueError, "does not match shared selection"):
            load_run(self.root)


if __name__ == "__main__":
    unittest.main()
