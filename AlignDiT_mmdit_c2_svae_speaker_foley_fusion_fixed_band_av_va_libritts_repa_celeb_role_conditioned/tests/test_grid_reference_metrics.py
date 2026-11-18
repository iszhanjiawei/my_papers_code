"""CPU-only contract tests; no neural models, GPU or real audio are loaded."""

import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from aligndit.script.eval.eval_celebvdub_grid_reference import (
    EXPECTED_COUNT,
    attach_identities,
    build_test_set,
    load_manifest,
    make_summary,
    parse_args,
    run,
    sha256_file,
    validate_inference_summary,
    word_edit_counts,
)


class GridReferenceMetricTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.reference = self.root / "grid.wav"
        self.target = self.root / "target.wav"
        self.reference.touch()
        self.target.touch()
        # All 213 examples deliberately share a basename, but not target_id.
        self.rows = [
            {
                "pair_id": f"pair-{i}",
                "target_id": f"test/video{i}/0_0",
                "target_text": "hello world",
                "target_gt_audio": str(self.target),
                "ref_id": f"grid/s{1 + i % 2}/clip{i}",
                "ref_speaker_id": f"s{1 + i % 2}",
                "ref_audio": str(self.reference),
            }
            for i in range(EXPECTED_COUNT)
        ]
        self.manifest = self.root / "pairs.jsonl"
        self.generated = self.root / "generated"
        for row in self.rows:
            path = self.generated / f"{row['target_id']}.wav"
            path.parent.mkdir(parents=True)
            with wave.open(str(path), "wb") as handle:
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(16000)
                handle.writeframes(b"\x00\x00" * 32)

    def write_manifest(self, rows):
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def write_inference_summary(self):
        outputs = []
        for row in self.rows:
            output = {key: row[key] for key in ("pair_id", "target_id", "ref_id", "ref_speaker_id")}
            relative = f"{row['target_id']}.wav"
            output.update(relative_path=relative, samples=32, sha256=sha256_file(self.generated / relative))
            outputs.append(output)
        summary = {
            "protocol": "celebvdub_grid_reference_one_per_target_v1",
            "pair_manifest_sha256": sha256_file(self.manifest),
            "count": EXPECTED_COUNT, "partial_smoke_test": False, "outputs": outputs,
        }
        (self.generated / "inference_summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return summary

    def test_same_basename_targets_remain_distinct(self):
        self.write_manifest(self.rows)
        records = load_manifest(self.manifest)
        results = [{"wav": "0_0", "sim": i / EXPECTED_COUNT} for i in range(EXPECTED_COUNT)]
        identified = attach_identities(records, results, "sim")
        self.assertEqual(len({row["target_id"] for row in identified}), EXPECTED_COUNT)
        self.assertEqual(len({row["wav"] for row in identified}), EXPECTED_COUNT)
        self.assertEqual(identified[100]["pair_id"], "pair-100")
        self.assertEqual(identified[100]["sim"], 100 / EXPECTED_COUNT)
        self.assertEqual(set(make_summary(identified, "sim")["per_reference_speaker"]), {"s1", "s2"})

    def test_speaker_and_emotion_use_different_references(self):
        speaker = build_test_set(self.rows, self.generated, "sim")
        emotion = build_test_set(self.rows, self.generated, "emosim")
        embed = build_test_set(self.rows, self.generated, "emoembed")
        wer = build_test_set(self.rows, self.generated, "wer")
        self.assertEqual(speaker[0][1], str(self.reference))
        self.assertEqual(emotion[0][1], str(self.target))
        self.assertEqual(embed[0][1], str(self.target))
        self.assertEqual(wer[0][2], "hello world")

    def test_corpus_wer_is_not_sentence_mean(self):
        results = [
            {"wav": "0_0", "wer": 1.0, "truth": "hello", "hypo": "wrong"},
            {"wav": "0_0", "wer": 0.0, "truth": "one two three four five six seven eight nine", "hypo": "one two three four five six seven eight nine"},
        ]
        identified = attach_identities(self.rows[:2], results, "wer")
        summary = make_summary(identified, "wer")
        self.assertEqual(summary["wer"], 0.1)
        self.assertNotEqual(summary["wer"], 0.5)
        self.assertEqual(summary["word_edit_distance"], 1)
        self.assertEqual(summary["reference_word_count"], 10)
        self.assertEqual(summary["display_value"], "0.10000")
        self.assertEqual(word_edit_counts("a b", "x a b c"), (2, 2))

    def test_missing_records_and_files_fail(self):
        self.write_manifest(self.rows[:-1])
        with self.assertRaisesRegex(ValueError, "exactly 213"):
            load_manifest(self.manifest)
        self.write_manifest(self.rows)
        (self.generated / f"{self.rows[-1]['target_id']}.wav").unlink()
        with self.assertRaises(FileNotFoundError):
            build_test_set(self.rows, self.generated, "sim")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            attach_identities(self.rows, [], "sim")
        self.reference.unlink()
        with self.assertRaises(FileNotFoundError):
            load_manifest(self.manifest)

    def test_duplicate_identity_and_unsafe_path_fail(self):
        self.rows[-1]["pair_id"] = self.rows[0]["pair_id"]
        self.write_manifest(self.rows)
        with self.assertRaisesRegex(ValueError, "unique pair_id"):
            load_manifest(self.manifest)
        self.rows[-1]["pair_id"] = "unique-last"
        self.rows[-1]["target_id"] = "test/../escape"
        self.write_manifest(self.rows)
        with self.assertRaisesRegex(ValueError, "Invalid test"):
            load_manifest(self.manifest)

    def test_nonfinite_or_wrong_order_results_fail(self):
        with self.assertRaisesRegex(ValueError, "Invalid sim"):
            attach_identities(self.rows[:1], [{"wav": "0_0", "sim": float("nan")}], "sim")
        with self.assertRaisesRegex(ValueError, "order mismatch"):
            attach_identities(self.rows[:1], [{"wav": "wrong", "sim": 0.5}], "sim")

    def test_complete_mock_evaluation_writes_ids_and_summary(self):
        self.write_manifest(self.rows)
        self.write_inference_summary()
        args = parse_args([
            "--manifest", str(self.manifest), "--gen-wav-dir", str(self.generated),
            "-e", "sim", "--wavlm-ckpt", str(self.reference),
            "--wavlm-base-ckpt", str(self.reference),
        ])
        results = [{"wav": "0_0", "sim": 0.75} for _ in self.rows]
        with patch("aligndit.script.eval.eval_celebvdub_grid_reference.run_worker", return_value=results) as worker:
            summary = run(args)
        self.assertEqual(worker.call_args[0][1][0][1], str(self.reference))
        self.assertEqual(summary["count"], EXPECTED_COUNT)
        self.assertEqual(summary["sim"], 0.75)
        lines = (self.generated / "_sim_results.jsonl").read_text().splitlines()
        self.assertEqual(lines[-1], "SIM: 0.75000")
        records = [json.loads(line) for line in lines if line.startswith("{")]
        self.assertEqual(len({row["pair_id"] for row in records}), EXPECTED_COUNT)
        saved_summary = json.loads((self.generated / "_sim_summary.json").read_text())
        self.assertEqual(saved_summary["manifest_sha256"], summary["manifest_sha256"])
        with self.assertRaises(FileExistsError):
            run(args)

    def test_inference_manifest_and_audio_must_match(self):
        self.write_manifest(self.rows)
        with self.assertRaises(FileNotFoundError):
            validate_inference_summary(self.rows, self.manifest, self.generated)
        summary = self.write_inference_summary()
        validate_inference_summary(self.rows, self.manifest, self.generated)
        summary_path = self.generated / "inference_summary.json"
        for field, value in (("pair_manifest_sha256", "wrong"), ("count", 212), ("partial_smoke_test", True)):
            modified = dict(summary)
            modified[field] = value
            summary_path.write_text(json.dumps(modified), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete 213-pair GRID"):
                validate_inference_summary(self.rows, self.manifest, self.generated)
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        (self.generated / f"{self.rows[0]['target_id']}.wav").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            validate_inference_summary(self.rows, self.manifest, self.generated)


if __name__ == "__main__":
    unittest.main()
