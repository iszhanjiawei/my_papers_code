import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from f5_tts.model import trainer as trainer_module
from f5_tts.model.trainer import Trainer


class _Stateful:
    def __init__(self, value: int):
        self.value = value

    def state_dict(self):
        return {"value": torch.tensor(self.value)}


class _FakeAccelerator:
    def __init__(self, *, is_main: bool, fail_save: bool = False):
        self.is_main_process = is_main
        self.fail_save = fail_save
        self.barrier_count = 0
        self.save_count = 0

    def wait_for_everyone(self):
        self.barrier_count += 1

    @staticmethod
    def unwrap_model(model):
        return model

    def save(self, checkpoint, destination):
        self.save_count += 1
        if self.fail_save:
            destination.write(b"incomplete checkpoint")
            destination.flush()
            raise RuntimeError("injected serialization failure")
        torch.save(checkpoint, destination)


def _make_trainer(checkpoint_path: Path, accelerator: _FakeAccelerator) -> Trainer:
    trainer = Trainer.__new__(Trainer)
    trainer.accelerator = accelerator
    trainer.model = _Stateful(1)
    trainer.optimizer = _Stateful(2)
    trainer.ema_model = _Stateful(3)
    trainer.scheduler = _Stateful(4)
    trainer.checkpoint_path = str(checkpoint_path)
    trainer.keep_last_n_checkpoints = -1
    return trainer


class AtomicCheckpointTest(unittest.TestCase):
    def test_success_replaces_last_checkpoint_and_synchronizes_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            destination = checkpoint_path / "model_last.pt"
            destination.write_bytes(b"previous complete checkpoint")
            accelerator = _FakeAccelerator(is_main=True)
            trainer = _make_trainer(checkpoint_path, accelerator)
            events = []
            real_fsync = trainer_module.os.fsync
            real_replace = trainer_module.os.replace

            def observed_fsync(descriptor):
                events.append(("fsync", stat.S_ISDIR(trainer_module.os.fstat(descriptor).st_mode)))
                return real_fsync(descriptor)

            def observed_replace(source, target):
                events.append(("replace", source, target))
                return real_replace(source, target)

            with (
                patch.object(trainer_module.os, "fsync", side_effect=observed_fsync),
                patch.object(trainer_module.os, "replace", side_effect=observed_replace),
            ):
                trainer.save_checkpoint(update=500, last=True)

            checkpoint = torch.load(destination, map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["update"], 500)
            self.assertEqual(accelerator.save_count, 1)
            self.assertEqual(accelerator.barrier_count, 2)
            self.assertEqual(list(checkpoint_path.glob("*.tmp")), [])
            self.assertEqual([event[0] for event in events], ["fsync", "replace", "fsync"])
            self.assertFalse(events[0][1])
            self.assertTrue(events[2][1])

    def test_failed_serialization_preserves_previous_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            destination = checkpoint_path / "model_last.pt"
            previous = b"previous complete checkpoint"
            destination.write_bytes(previous)
            accelerator = _FakeAccelerator(is_main=True, fail_save=True)
            trainer = _make_trainer(checkpoint_path, accelerator)

            with self.assertRaisesRegex(RuntimeError, "injected serialization failure"):
                trainer.save_checkpoint(update=500, last=True)

            self.assertEqual(destination.read_bytes(), previous)
            self.assertEqual(accelerator.barrier_count, 1)
            self.assertEqual(list(checkpoint_path.glob("*.tmp")), [])

    def test_non_main_rank_only_participates_in_both_barriers(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            accelerator = _FakeAccelerator(is_main=False)
            trainer = _make_trainer(checkpoint_path, accelerator)

            trainer.save_checkpoint(update=500, last=True)

            self.assertEqual(accelerator.save_count, 0)
            self.assertEqual(accelerator.barrier_count, 2)
            self.assertFalse((checkpoint_path / "model_last.pt").exists())

    def test_disabled_numbered_checkpoint_does_not_skip_second_barrier(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory)
            accelerator = _FakeAccelerator(is_main=True)
            trainer = _make_trainer(checkpoint_path, accelerator)
            trainer.keep_last_n_checkpoints = 0

            trainer.save_checkpoint(update=500, last=False)

            self.assertEqual(accelerator.save_count, 0)
            self.assertEqual(accelerator.barrier_count, 2)
            self.assertEqual(list(checkpoint_path.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
