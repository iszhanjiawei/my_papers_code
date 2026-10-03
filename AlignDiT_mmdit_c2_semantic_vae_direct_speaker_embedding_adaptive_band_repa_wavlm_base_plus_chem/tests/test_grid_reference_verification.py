"""CPU integrity tests; fixtures include two targets with the same basename."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from aligndit.script.eval import verify_grid_reference_results as verifier


class GridReferenceVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "outputs"
        self.output.mkdir()
        self.manifest = self.root / "pairs.jsonl"
        self.checkpoint = self.root / "model_150000.pt"
        self.checkpoint.write_bytes(b"fixture checkpoint, not a model")
        self.pairs = []
        generated = []
        for index in range(2):
            target = f"test/video{index}/same_clip"
            pair = {
                "pair_id": f"pair{index}", "target_id": target,
                "ref_id": f"grid/s{index + 1}/utterance", "ref_speaker_id": f"s{index + 1}",
                "target_text": "One." if index == 0 else "one two three",
                "target_num_samples": 400,
            }
            for field, suffix in (("ref_audio", ".wav"), ("ref_latent_path", ".npy"),
                                  ("ref_speaker_path", ".npy"), ("target_video_path", ".npy")):
                path = self.root / f"{index}_{field}{suffix}"
                if suffix == ".wav":
                    sf.write(path, np.zeros(400, dtype=np.float32), 16000)
                else:
                    np.save(path, np.ones((2, 3), dtype=np.float32))
                pair[field] = str(path)
                hash_field = field.removesuffix("_path") + "_sha256"
                pair[hash_field] = verifier.sha256_file(path)
            target_gt = self.root / f"target_gt_{index}.wav"
            sf.write(target_gt, np.zeros(400, dtype=np.float32), 16000)
            pair["target_gt_audio"] = str(target_gt)
            wav = self.output / (target + ".wav")
            wav.parent.mkdir(parents=True)
            sf.write(wav, np.zeros(400, dtype=np.float32), 16000)
            feature = self.output / "avhubert_feat" / (target + ".npy")
            feature.parent.mkdir(parents=True)
            np.save(feature, np.ones((2, 3), dtype=np.float32))
            generated.append({**{key: pair[key] for key in verifier.IDENTITY_FIELDS},
                              "relative_path": target + ".wav", "samples": 400,
                              "sha256": verifier.sha256_file(wav)})
            self.pairs.append(pair)
        self.write_rows(self.manifest, self.pairs)
        manifest_hash = verifier.sha256_file(self.manifest)
        inference = {
            "count": 2, "partial_smoke_test": False, "pair_manifest": str(self.manifest),
            "pair_manifest_sha256": manifest_hash, "protocol": "fixture",
            "checkpoint": {"path": str(self.checkpoint), "update": 150000, "weights": "EMA",
                           "sha256": verifier.sha256_file(self.checkpoint)},
            "generation": {"target_acoustic_inputs": False}, "outputs": generated,
        }
        self.write_json(self.output / "inference_summary.json", inference)
        for task in verifier.REQUIRED_TASKS:
            rows = []
            for index, pair in enumerate(self.pairs):
                row = {**{key: pair[key] for key in verifier.IDENTITY_FIELDS}, "wav": pair["target_id"],
                       "ref_audio": pair["ref_audio"], "target_gt_audio": pair["target_gt_audio"],
                       task: 0.4 + 0.2 * index}
                if task == "wer":
                    row.update(raw_truth=pair["target_text"], truth=verifier.normalized_wer_text(pair["target_text"]),
                               raw_hypo="wrong" if index == 0 else "one two three",
                               hypo="wrong" if index == 0 else "one two three",
                               wer=1.0 if index == 0 else 0.0, word_edit_distance=1 if index == 0 else 0,
                               reference_word_count=1 if index == 0 else 3)
                rows.append(row)
            summary = {**verifier.aggregate(rows, task), "manifest_sha256": manifest_hash,
                       "gen_wav_dir": str(self.output), "task": task,
                       "per_reference_speaker": {row["ref_speaker_id"]: verifier.aggregate([row], task) for row in rows}}
            self.write_rows(self.output / f"_{task}_results.jsonl", rows,
                            trailer=f"\n{task.upper()}: {summary['display_value']}\n")
            self.write_json(self.output / f"_{task}_summary.json", summary)
        self.args = argparse.Namespace(manifest=self.manifest, output_dir=self.output, expected_step=150000)

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    @staticmethod
    def write_rows(path, rows, trailer=""):
        path.write_text("".join(json.dumps(row) + "\n" for row in rows) + trailer, encoding="utf-8")

    def verify(self):
        with patch.object(verifier, "EXPECTED_COUNT", 2), patch("builtins.print"):
            return verifier.verify(self.args)

    def test_success_full_ids_corpus_wer_and_idempotence(self):
        result = self.verify()
        self.assertEqual(result["metrics"]["wer"]["wer"], 0.25)
        self.assertEqual(result["metrics"]["sim"]["display_value"], "0.50000")
        self.assertEqual(self.verify(), result)

    def test_reordered_same_basename_results_fail(self):
        path = self.output / "_sim_results.jsonl"
        rows = verifier.read_rows(path, task="sim")
        self.write_rows(path, rows[::-1])
        with self.assertRaisesRegex(ValueError, "identity/order"):
            self.verify()

    def test_missing_sample_fails(self):
        path = self.output / "_wer_results.jsonl"
        self.write_rows(path, verifier.read_rows(path, task="wer")[:1])
        with self.assertRaisesRegex(ValueError, "Expected 2 wer"):
            self.verify()

    def test_changed_reference_fails(self):
        Path(self.pairs[0]["ref_audio"]).write_bytes(b"changed reference")
        with self.assertRaisesRegex(ValueError, "Changed artifact"):
            self.verify()

    def test_nonfinite_feature_fails(self):
        path = self.output / "avhubert_feat" / (self.pairs[0]["target_id"] + ".npy")
        np.save(path, np.array([[np.nan]], dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "Invalid generated AV"):
            self.verify()

    def test_extra_wav_fails(self):
        sf.write(self.output / "extra.wav", np.zeros(400), 16000)
        with self.assertRaisesRegex(ValueError, "Actual WAV file set"):
            self.verify()

    def test_utterance_mean_wer_summary_fails(self):
        path = self.output / "_wer_summary.json"
        summary = verifier.read_object(path)
        summary["wer"] = 0.5
        self.write_json(path, summary)
        with self.assertRaisesRegex(ValueError, "Recomputed value mismatch"):
            self.verify()


if __name__ == "__main__":
    unittest.main()
