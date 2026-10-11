"""CPU regression checks for parallel ownership and completion guarantees."""

import fcntl
import json
import signal
import subprocess
import tempfile
import unittest
from itertools import chain
from pathlib import Path
from unittest.mock import patch

from aligndit.script.misc import prepare_grid_parallel as parallel


class GridParallelShardTests(unittest.TestCase):
    def test_gpu_workers_per_device_defaults_to_one_and_accepts_explicit_fanout(self):
        self.assertEqual(parallel.parse_args([]).gpu_workers_per_device, 1)
        self.assertEqual(parallel.parse_args(["--gpu-workers-per-device", "2"]).gpu_workers_per_device, 2)

    def test_shards_are_deterministic_disjoint_and_cover_inventory_exactly(self):
        for total in (0, 1, 2, 7, 23):
            for count in (1, 2, 4, 12):
                with self.subTest(total=total, count=count):
                    shards = [parallel.shard_indices(total, index, count) for index in range(count)]
                    self.assertEqual(shards, [
                        parallel.shard_indices(total, index, count) for index in range(count)
                    ])
                    combined = list(chain.from_iterable(shards))
                    self.assertEqual(sorted(combined), list(range(total)))
                    self.assertEqual(len(combined), len(set(combined)))
                    for index, shard in enumerate(shards):
                        self.assertEqual(shard, list(range(index, total, count)))
                    sizes = [len(shard) for shard in shards]
                    self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_invalid_shard_requests_fail_before_any_work(self):
        for request in ((-1, 0, 1), (5, 0, 0), (5, 0, -1), (5, -1, 2), (5, 2, 2)):
            with self.subTest(request=request), self.assertRaises(ValueError):
                parallel.shard_indices(*request)


class _FakeProcess:
    """A terminated leader and a running sibling, without touching real PIDs."""

    def __init__(self, pid, returncode=None):
        self.pid = pid
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake-worker", timeout)
        return self.returncode


class GridParallelPoolTests(unittest.TestCase):
    def test_worker_failure_terminates_only_owned_groups_including_exited_leader(self):
        with tempfile.TemporaryDirectory(prefix="grid-parallel-pool-") as directory:
            root = Path(directory)
            commands = [
                parallel.WorkerCommand("audio-failed", ["fake-worker", "0"], root / "failed.log"),
                parallel.WorkerCommand("audio-running", ["fake-worker", "1"], root / "running.log"),
            ]
            failed = _FakeProcess(901001, 7)
            running = _FakeProcess(901002)
            owned = {process.pid: process for process in (failed, running)}
            signals = []

            def signal_owned_group(pgid, requested_signal):
                self.assertIn(pgid, owned, "cleanup must never signal a group it did not start")
                signals.append((pgid, requested_signal))
                if owned[pgid].returncode is None:
                    owned[pgid].returncode = -requested_signal

            with (
                patch.object(parallel.subprocess, "Popen", side_effect=[failed, running]) as start,
                patch.object(parallel.os, "killpg", side_effect=signal_owned_group),
                self.assertRaises(RuntimeError),
            ):
                parallel.run_pool(commands, poll_seconds=0.001, terminate_grace=0)

            self.assertEqual(start.call_count, 2)
            for invocation in start.call_args_list:
                self.assertTrue(invocation.kwargs["start_new_session"])
            terminated = {pgid for pgid, requested_signal in signals if requested_signal == signal.SIGTERM}
            self.assertEqual(terminated, set(owned))


class GridParallelCoordinatorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="grid-parallel-parent-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "cache"
        self.root.mkdir()
        self.args = parallel.parse_args([])
        self.args.output_dir = self.root
        self.args.gpus = ("0", "1")
        self.args.audio_workers = 2
        self.args.audio_threads = 1
        self.args.torch_threads = 1
        self.rows = [
            {"utterance_key": "grid/s1/bbaf2n", "split": "train"},
            {"utterance_key": "grid/s2/bbaf3s", "split": "val"},
        ]
        self.metadata = {
            "schema_version": 1,
            "dataset": "GRID",
            "selection": {"mode": "full"},
            "split_counts": {"train": 1, "val": 1},
            "feature_contract": {},
            "extraction_sha256": "frozen-extraction",
            "extraction_protocol": {"implementation_sha256": parallel.prep.implementation_hashes()},
            "manifests": {},
        }
        manifests = self.root / "manifests"
        manifests.mkdir()
        for name, rows in (("inventory", self.rows), ("train", self.rows[:1]), ("val", self.rows[1:])):
            path = manifests / f"{name}.jsonl"
            path.write_bytes(parallel.prep.jsonl_bytes(rows))
            self.metadata["manifests"][path.name] = {
                "count": len(rows), "size_bytes": path.stat().st_size,
                "sha256": parallel.prep.sha256_file(path),
            }
        (manifests / "inventory_meta.json").write_text(json.dumps(self.metadata))
        self.plan = (self.rows, self.metadata, {})
        self.complete = self.root / "complete.json"
        self.guard = self.root / "WRITE_ACTIVE.json"

    def assert_parent_owns_cache(self):
        self.assertTrue(self.guard.is_file())
        self.assertFalse(self.complete.exists())
        with (self.root / ".prepare.lock").open("a") as competing_lock, self.assertRaises(BlockingIOError):
            fcntl.flock(competing_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_gpu_fanout_uses_global_disjoint_shards_and_caps_workers_to_rows(self):
        self.args.gpus = ("2", "3")
        self.args.gpu_workers_per_device = 2
        for total, expected_masks in ((7, ["2", "2", "3", "3"]), (3, ["2", "2", "3"])):
            with self.subTest(total=total):
                rows = [dict(self.rows[index % 2], utterance_key=f"grid/s{index}/bbaf2n")
                        for index in range(total)]
                metadata = json.loads(json.dumps(self.metadata))
                metadata["split_counts"] = {
                    split: sum(row["split"] == split for row in rows) for split in ("train", "val")
                }
                for name in ("inventory", "train", "val"):
                    selected = rows if name == "inventory" else [row for row in rows if row["split"] == name]
                    path = self.root / f"manifests/{name}.jsonl"
                    path.write_bytes(parallel.prep.jsonl_bytes(selected))
                    metadata["manifests"][path.name] = {
                        "count": len(selected), "size_bytes": path.stat().st_size,
                        "sha256": parallel.prep.sha256_file(path),
                    }
                (self.root / "manifests/inventory_meta.json").write_text(json.dumps(metadata))
                with (
                    patch.object(parallel.prep, "reuse_plan", return_value=(rows, metadata, {})),
                    patch.object(parallel, "run_pool", return_value=[]) as pool,
                    patch.object(parallel.prep, "verify", return_value=True),
                ):
                    parallel.coordinate(self.args)

                self.assertEqual(pool.call_count, 2)
                commands = pool.call_args_list[1].args[0]
                self.assertEqual([command.env["CUDA_VISIBLE_DEVICES"] for command in commands], expected_masks)
                shards = []
                for index, command in enumerate(commands):
                    argv = command.command
                    self.assertEqual(int(argv[argv.index("--shard-index") + 1]), index)
                    count = int(argv[argv.index("--shard-count") + 1])
                    self.assertEqual(count, len(expected_masks))
                    shards.append(parallel.shard_indices(total, index, count))
                combined = list(chain.from_iterable(shards))
                self.assertEqual(sorted(combined), list(range(total)))
                self.assertEqual(len(combined), len(set(combined)))

    def test_nonpositive_gpu_workers_per_device_fails_before_plan_or_workers(self):
        for count in (0, -1):
            with self.subTest(count=count):
                self.args.gpu_workers_per_device = count
                with (
                    patch.object(parallel.prep, "reuse_plan") as reuse,
                    patch.object(parallel, "run_pool") as pool,
                    patch.object(parallel.prep, "verify") as verify,
                    self.assertRaisesRegex(ValueError, "gpu-workers-per-device"),
                ):
                    parallel.coordinate(self.args)
                reuse.assert_not_called()
                pool.assert_not_called()
                verify.assert_not_called()
                self.assertFalse(self.guard.exists())
                self.assertFalse(self.complete.exists())

    def test_failed_audio_pool_prevents_gpu_launch_and_final_verification(self):
        self.complete.write_text('{"complete": true}')

        def fail_audio(commands, **kwargs):
            self.assert_parent_owns_cache()
            raise RuntimeError("audio worker failed")

        with (
            patch.object(parallel.prep, "reuse_plan", return_value=self.plan),
            patch.object(parallel, "run_pool", side_effect=fail_audio) as pool,
            patch.object(parallel.prep, "verify") as verify,
            self.assertRaisesRegex(RuntimeError, "audio worker failed"),
        ):
            parallel.coordinate(self.args)

        self.assertEqual(pool.call_count, 1)
        self.assertEqual(len(pool.call_args.args[0]), self.args.audio_workers)
        verify.assert_not_called()
        self.assertFalse(self.guard.exists())
        self.assertFalse(self.complete.exists())

    def test_success_retains_parent_guard_until_full_verification_publishes_completion(self):
        self.complete.write_text('{"complete": true, "stale": true}')
        events = []

        def finish_pool(commands, **kwargs):
            self.assert_parent_owns_cache()
            events.append("pool")
            return []

        def verify_all(rows, metadata, args, *, required):
            self.assert_parent_owns_cache()
            self.assertEqual(events, ["pool", "pool"])
            self.assertEqual(rows, self.rows)
            self.assertEqual(metadata, self.metadata)
            self.assertTrue(required)
            self.complete.write_text('{"complete": true, "verified": true}')
            events.append("verify")
            return True

        with (
            patch.object(parallel.prep, "reuse_plan", return_value=self.plan),
            patch.object(parallel, "run_pool", side_effect=finish_pool),
            patch.object(parallel.prep, "verify", side_effect=verify_all) as verify,
        ):
            parallel.coordinate(self.args)

        verify.assert_called_once()
        self.assertEqual(events, ["pool", "pool", "verify"])
        self.assertFalse(self.guard.exists())
        self.assertEqual(json.loads(self.complete.read_text()), {"complete": True, "verified": True})
        with (self.root / ".prepare.lock").open("a") as released_lock:
            fcntl.flock(released_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_verification_failure_clears_guard_without_publishing_completion(self):
        with (
            patch.object(parallel.prep, "reuse_plan", return_value=self.plan),
            patch.object(parallel, "run_pool", return_value=[]),
            patch.object(parallel.prep, "verify", side_effect=RuntimeError("invalid receipt")) as verify,
            self.assertRaisesRegex(RuntimeError, "invalid receipt"),
        ):
            parallel.coordinate(self.args)

        verify.assert_called_once()
        self.assertTrue(verify.call_args.kwargs["required"])
        self.assertFalse(self.guard.exists())
        self.assertFalse(self.complete.exists())

    def test_competing_parent_cannot_change_guard_or_completion(self):
        self.complete.write_text("existing completion")
        self.guard.write_text("existing parent guard")
        with (self.root / ".prepare.lock").open("a") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with (
                patch.object(parallel.prep, "reuse_plan") as reuse,
                patch.object(parallel, "run_pool") as pool,
                patch.object(parallel.prep, "verify") as verify,
                self.assertRaises(BlockingIOError),
            ):
                parallel.coordinate(self.args)

        reuse.assert_not_called()
        pool.assert_not_called()
        verify.assert_not_called()
        self.assertEqual(self.complete.read_text(), "existing completion")
        self.assertEqual(self.guard.read_text(), "existing parent guard")

    def test_changed_extractor_source_prevents_completion_after_workers_exit(self):
        changed = dict(self.metadata["extraction_protocol"]["implementation_sha256"])
        changed["script/misc/prepare_grid.py"] = "changed-after-launch"
        with (
            patch.object(parallel.prep, "reuse_plan", return_value=self.plan),
            patch.object(parallel, "run_pool", return_value=[]),
            patch.object(parallel.prep, "implementation_hashes", return_value=changed),
            patch.object(parallel.prep, "verify") as verify,
            self.assertRaisesRegex(RuntimeError, "source changed"),
        ):
            parallel.coordinate(self.args)

        verify.assert_not_called()
        self.assertFalse(self.guard.exists())
        self.assertFalse(self.complete.exists())

    def _make_worker_request(self):
        self.args.semantic_vae_spec = self.root / "spec.json"
        self.args.semantic_vae_spec.write_text("{}")
        request = {
            "owner_pid": 900001,
            "run_id": "owned-run",
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(self.args).items()
            },
            "count": len(self.rows),
            "inventory_sha256": self.metadata["manifests"]["inventory.jsonl"]["sha256"],
            "inventory_meta_sha256": parallel.prep.sha256_file(self.root / "manifests/inventory_meta.json"),
            "extraction_sha256": self.metadata["extraction_sha256"],
            "coordinator_sha256": parallel.prep.sha256_file(Path(parallel.__file__)),
        }
        config = self.root / "worker_request.json"
        config.write_text(json.dumps(request))
        self.guard.write_text(json.dumps({"pid": request["owner_pid"], "run_id": request["run_id"]}))
        return config, request

    def test_audio_worker_extracts_only_its_shard_and_cannot_publish_completion(self):
        config, request = self._make_worker_request()
        original_guard = self.guard.read_bytes()
        with (
            patch.object(parallel, "_parent_death_signal"),
            patch.object(parallel.os, "getppid", return_value=request["owner_pid"]),
            patch.object(parallel.prep, "run_stage") as run_stage,
            patch.object(parallel.prep, "verify") as verify,
        ):
            parallel.worker(config, "audio", shard_index=1, shard_count=2)

        run_stage.assert_called_once()
        stage, rows, metadata, _spec, args = run_stage.call_args.args
        self.assertEqual(stage, "audio")
        self.assertEqual(rows, self.rows[1:])
        self.assertEqual(metadata, self.metadata)
        self.assertEqual(args.device, "cpu")
        verify.assert_not_called()
        self.assertFalse(self.complete.exists())
        self.assertEqual(self.guard.read_bytes(), original_guard)

    def test_stale_or_orphan_worker_cannot_write_any_artifacts(self):
        for wrong_parent in (False, True):
            with self.subTest(wrong_parent=wrong_parent):
                config, request = self._make_worker_request()
                parent = request["owner_pid"]
                if wrong_parent:
                    parent += 1
                else:
                    self.guard.write_text(json.dumps({"pid": parent, "run_id": "another-run"}))
                original_guard = self.guard.read_bytes()
                with (
                    patch.object(parallel, "_parent_death_signal"),
                    patch.object(parallel.os, "getppid", return_value=parent),
                    patch.object(parallel.prep, "run_stage") as run_stage,
                    patch.object(parallel.prep, "verify") as verify,
                    self.assertRaises(RuntimeError),
                ):
                    parallel.worker(config, "audio", shard_index=0, shard_count=2)

                run_stage.assert_not_called()
                verify.assert_not_called()
                self.assertFalse(self.complete.exists())
                self.assertEqual(self.guard.read_bytes(), original_guard)


if __name__ == "__main__":
    unittest.main()
