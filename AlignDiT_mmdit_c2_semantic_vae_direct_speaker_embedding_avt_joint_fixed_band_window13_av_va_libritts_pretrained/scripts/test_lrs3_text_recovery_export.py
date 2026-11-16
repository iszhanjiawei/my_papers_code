#!/usr/bin/env python3
"""Regression tests for the immutable text-release output hash contract."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from lrs3_text_recovery_export import REQUIRED_RELEASE_FILES, frozen_output_hashes


class FrozenOutputHashesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name)
        self.summary = {"processing_complete": True, "pending_segments": 0}
        for name in REQUIRED_RELEASE_FILES:
            payload = b"" if name == "labels.pending.jsonl" else b'{"example": true}\n'
            (self.output / name).write_bytes(payload)
        (self.output / "summary.json").write_text(json.dumps(self.summary) + "\n")

    def test_all_json_and_jsonl_outputs_are_bound_to_exact_bytes(self):
        (self.output / "export.log").write_text("not an output contract\n")
        (self.output / "export.lock").write_text("")
        hashes = frozen_output_hashes(self.output, self.summary)
        self.assertEqual(set(hashes), REQUIRED_RELEASE_FILES)
        for name, value in hashes.items():
            self.assertEqual(value, hashlib.sha256((self.output / name).read_bytes()).hexdigest())
        self.assertEqual(hashes["labels.pending.jsonl"], hashlib.sha256(b"").hexdigest())

    def test_missing_any_required_output_is_rejected(self):
        for name in REQUIRED_RELEASE_FILES:
            path = self.output / name
            payload = path.read_bytes()
            path.unlink()
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "missing required outputs"):
                frozen_output_hashes(self.output, self.summary)
            path.write_bytes(payload)

    def test_modified_label_bytes_change_the_bound_hash(self):
        before = frozen_output_hashes(self.output, self.summary)
        path = self.output / "labels.strict.accepted.jsonl"
        path.write_bytes(path.read_bytes() + b'{"another": true}\n')
        after = frozen_output_hashes(self.output, self.summary)
        self.assertNotEqual(before[path.name], after[path.name])
        self.assertEqual(after[path.name], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_summary_race_and_incomplete_release_are_rejected(self):
        for replacement in [{"processing_complete": False, "pending_segments": 0},
                            {"processing_complete": True, "pending_segments": 1}]:
            (self.output / "summary.json").write_text(json.dumps(replacement))
            with self.subTest(summary=replacement), self.assertRaisesRegex(ValueError, "summary changed"):
                frozen_output_hashes(self.output, self.summary)
            with self.subTest(incomplete=replacement), self.assertRaisesRegex(ValueError, "incomplete"):
                frozen_output_hashes(self.output, replacement)

    def test_nonempty_pending_file_is_rejected_even_if_summary_claims_zero(self):
        (self.output / "labels.pending.jsonl").write_text('{"sample_id": "pending"}\n')
        with self.assertRaisesRegex(ValueError, "containing pending labels"):
            frozen_output_hashes(self.output, self.summary)

    def test_existing_receipt_is_not_included_in_its_own_hashes(self):
        (self.output / "frozen_export.json").write_text('{"files": {}}\n')
        self.assertNotIn("frozen_export.json", frozen_output_hashes(self.output, self.summary))


if __name__ == "__main__":
    unittest.main()
