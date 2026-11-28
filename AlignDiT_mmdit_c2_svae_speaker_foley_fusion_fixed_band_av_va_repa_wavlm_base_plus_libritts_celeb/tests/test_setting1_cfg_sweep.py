"""CPU-only orchestration checks; the full run also checks WAV hash invariance."""
from pathlib import Path
import tempfile
import unittest

from aligndit.script.eval import sweep_setting1_cfg as sweep


class Setting1SweepTests(unittest.TestCase):
    def test_numeric_check_accepts_only_tiny_native_drift(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.wav", Path(directory) / "b.wav"
            wave = np.sin(np.arange(1600) / 10) * 0.1
            sf.write(a, wave, 16000, subtype="PCM_16")
            sf.write(b, wave + 1e-5, 16000, subtype="PCM_16")
            self.assertFalse(sweep.compare_waveforms(a, b)["bitwise_equal"])
            sf.write(b, wave * 0.9, 16000, subtype="PCM_16")
            with self.assertRaises(AssertionError):
                sweep.compare_waveforms(a, b)

    def test_uses_this_project_and_fixed_native_parameters(self):
        self.assertEqual(sweep.PROJECT, Path(__file__).resolve().parents[1])
        for cfg in (3, 4, 5):
            args = sweep.inference_args(Path("output"), cfg, 1, 5)
            for key, value in (("--step", "150000"), ("--cfg-text", "5"),
                               ("--cfg-video", str(cfg)), ("--seed", "0"),
                               ("--nfe", "32"), ("--shard", "1"), ("--nshard", "5")):
                self.assertEqual(args[args.index(key) + 1], value)
            self.assertEqual(args[args.index("--config") + 1], sweep.CONFIG)

    def test_dispatch_success_and_failure_are_not_conflated(self):
        with tempfile.TemporaryDirectory() as directory:
            logs = Path(directory)
            sweep.dispatch([("ok", 0, ["-c", "print('verified')"], {})], logs, 1)
            self.assertIn("verified", (logs / "ok.log").read_text())
            with self.assertRaises(RuntimeError):
                sweep.dispatch([("bad", 1, ["-c", "raise SystemExit(2)"], {})], logs, 1)


if __name__ == "__main__":
    unittest.main()
