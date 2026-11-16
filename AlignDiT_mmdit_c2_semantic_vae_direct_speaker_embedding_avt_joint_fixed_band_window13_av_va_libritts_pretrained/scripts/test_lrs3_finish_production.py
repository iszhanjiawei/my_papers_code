"""Check finishing safety without running any model, cache or publication job."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from lrs3_finish_production import Finisher, PROTOCOL, cache_status, digest, write


class FinishTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.canonical = self.root / "manifests/source_inventory.jsonl"
        self.canonical.parent.mkdir(parents=True)
        self.rows = [{"sample_id": "a", "relative_stem": "train/pretrain/a", "original_clip_id": "pretrain/v/1",
                      "text_status": "original_pipeline"},
                     {"sample_id": "b", "relative_stem": "train/pretrain/b", "original_clip_id": "pretrain/v/2",
                      "text_status": "missing"}]
        self.canonical.write_text("".join(json.dumps(row) + "\n" for row in self.rows))
        self.fingerprint = digest(self.canonical)
        write(self.root / "reports/source_contract.json", {"source_inventory_sha256": self.fingerprint,
              "source_media_pairs": 2, "original_clip_count": 2})
        self.args = argparse.Namespace(root=self.root, name="finish", existing_wave=[], audio_run="audio",
            workers=8, finalizer_workers=16, poll_seconds=.001, wave_seconds=300, min_wave=10000)
        (self.root / "state/locks").mkdir(parents=True)
        self.lock = (self.root / "state/locks/finish.lock").open("a")
        self.addCleanup(self.lock.close)
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.finish = Finisher(self.args, self.lock.fileno())

    def cache_summary(self, **changes):
        return {"stage": "video40", "source_manifest": str(self.canonical), "manifest_sha256": self.fingerprint,
                "requested": 2, "succeeded": 2, "failed": 0, "complete": True, "finished_unix": 1, **changes}

    def test_false_completion_and_wrong_source_rejected(self):
        for changes in ({"succeeded": 1}, {"failed": 1}, {"requested": 1}, {"manifest_sha256": "wrong"},
                        {"finished_unix": None}, {"stage": "audio"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                cache_status(self.cache_summary(**changes), "video40", self.canonical, self.fingerprint, 2)

    def test_live_incomplete_waits_and_dead_or_finished_incomplete_fails(self):
        pending = self.cache_summary(complete=False, succeeded=0, finished_unix=None, pid=os.getpid())
        self.assertFalse(cache_status(pending, "video40", self.canonical, self.fingerprint, 2))
        for change in ({"pid": -100000}, {"finished_unix": 1}):
            with self.subTest(change=change), self.assertRaises(RuntimeError):
                cache_status({**pending, **change}, "video40", self.canonical, self.fingerprint, 2)

    def test_missing_audio_summary_rejected(self):
        with self.assertRaises(FileNotFoundError):
            self.finish.audio_done()

    def test_resume_checks_inputs_and_keeps_complete_stage(self):
        marker = self.root / "done"
        marker.write_text("ok")
        self.finish.state["stages"]["publication"] = {"status": "running", "pid": -1}
        self.finish.save()
        resumed = Finisher(self.args, self.lock.fileno())
        with mock.patch("lrs3_finish_production.subprocess.Popen") as popen:
            resumed.command("publication", ["must-not-run"], marker.exists)
        popen.assert_not_called()
        self.assertTrue(resumed.state["stages"]["publication"]["recovered_from_output"])
        self.canonical.write_text(self.canonical.read_text() + "{}\n")
        with self.assertRaises(ValueError):
            Finisher(self.args, self.lock.fileno())

    def test_incomplete_prior_attempt_never_restarts_a_writer(self):
        self.finish.state["stages"]["wave"] = {"status": "running", "pid": -1}
        with mock.patch("lrs3_finish_production.subprocess.Popen") as popen, self.assertRaises(RuntimeError):
            self.finish.command("wave", ["must-not-run"], lambda: False)
        popen.assert_not_called()

    def test_nonzero_child_preserves_log_and_failure(self):
        with self.assertRaises(RuntimeError):
            self.finish.command("bad", [sys.executable, "-c", "print('failure evidence');raise SystemExit(7)"], lambda: False)
        stage = self.finish.state["stages"]["bad"]
        self.assertEqual(stage["returncode"], 7)
        self.assertEqual(stage["status"], "failed")
        self.assertIn("failure evidence", Path(stage["log"]).read_text())

    def test_success_without_completion_artifact_rejected(self):
        with self.assertRaises(RuntimeError):
            self.finish.command("empty", [sys.executable, "-c", "pass"], lambda: False)
        self.assertEqual(self.finish.state["stages"]["empty"]["status"], "failed")

    def test_media_flag_requires_full_independent_coverage(self):
        directory = self.root / "reports/media_full_pipeline"
        summary = {"complete": True, "requested": 2, "verified": 2, "failed": 0, "original_ids": 2,
                   "complete_original_ids": 2, "protocol_sha256": PROTOCOL}
        write(directory / "summary.json", summary)
        write(directory / "state.json", {"status": "complete", "manifest": str(self.canonical),
              "manifest_sha256": self.fingerprint, "summary": summary})
        proof = directory / "independent_verification.jsonl"
        proof.write_text(json.dumps({**self.rows[0], "status": "ok"}) + "\n")
        with self.assertRaises(ValueError):
            self.finish.media_done()
        proof.write_text("".join(json.dumps({**row, "status": "ok"}) + "\n" for row in self.rows))
        self.assertTrue(self.finish.media_done())
        with proof.open("a") as f:
            f.write(json.dumps({**self.rows[1], "status": "failed"}) + "\n")
        with self.assertRaises(ValueError):
            self.finish.media_done()

    def test_dead_text_exporter_is_not_silent_wait(self):
        write(self.root / "text_recovery/release/summary.json", {"processing_complete": False})
        with self.assertRaises(RuntimeError):
            self.finish.text_done()

    def test_frozen_text_requires_exact_scope_and_hashes(self):
        release = self.root / "text_recovery/release"
        summary = {"processing_complete": True, "asr_and_requested_controls_complete": True,
                   "pending_segments": 0, "groups_requested": 1, "groups_complete": 1,
                   "segments_requested": 1, "strict_segments": 1, "pseudo_p1_segments": 0,
                   "pseudo_p2_candidate_segments": 0, "excluded_segments": 0}
        write(release / "summary.json", summary)
        for name in ("labels.strict.accepted.jsonl", "labels.pseudo_p1.accepted.jsonl"):
            (release / name).write_text("")
        frozen = {"source_manifest_sha256": self.fingerprint, "summary": summary,
                  "files": {p.name: digest(p) for p in release.iterdir()}}
        write(release / "frozen_export.json", frozen)
        self.assertTrue(self.finish.text_done())
        (release / "labels.strict.accepted.jsonl").write_text("changed")
        with self.assertRaises(ValueError):
            self.finish.text_done()
        frozen["summary"]["groups_complete"] = 0
        write(release / "frozen_export.json", frozen)
        with self.assertRaises(ValueError):
            self.finish.text_done()

    def test_final_pass_is_canonical_even_without_intermediate_waves(self):
        with mock.patch.object(self.finish, "audio_done", return_value=True), \
             mock.patch.object(self.finish, "media_done", return_value=True), \
             mock.patch.object(self.finish, "text_done", return_value=True), \
             mock.patch.object(self.finish, "quiet", return_value=True), \
             mock.patch.object(self.finish, "wave") as wave, \
             mock.patch.object(self.finish, "publish") as publish:
            self.finish.run_all()
        entry = wave.call_args.args[0]
        self.assertEqual(entry["manifest"], str(self.canonical))
        self.assertEqual(entry["count"], 2)
        self.assertEqual(entry["sha256"], self.fingerprint)
        publish.assert_called_once()
        self.assertEqual(self.finish.state["status"], "complete")

    def test_publication_commands_cannot_start_training(self):
        with mock.patch.object(self.finish, "command") as command:
            self.finish.publish()
        calls = command.call_args_list
        self.assertEqual([c.args[0] for c in calls], ["publish_full", "publish_full_pseudo", "dryrun_strict", "dryrun_pseudo"])
        for call in calls:
            self.assertNotIn("--start", call.args[1])
        self.assertIn("--strict-baseline", calls[1].args[1])


if __name__ == "__main__":
    unittest.main()
