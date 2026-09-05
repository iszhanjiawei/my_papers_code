"""CPU checks for shared physical masks and hidden-waveform isolation."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import torchaudio

from aligndit.script.eval.compare_native_audio_pretrains import (
    artifact,
    branch_name,
    create_attempt,
    keyed_seed,
    load_observed_input,
    physical_span,
    save_wave,
    sha256_file,
)


class PhysicalMaskTests(unittest.TestCase):
    def test_same_physical_span_maps_exactly_to_both_native_frame_rates(self):
        for samples in (64000, 65001, 95037, 159999, 160000):
            for key in ("dev-clean/1/2/example-a", "dev-other/3/4/example-b"):
                with self.subTest(samples=samples, key=key):
                    start, end = physical_span(samples, key)
                    self.assertGreaterEqual(start, 0)
                    self.assertGreater(end, start)
                    self.assertLessEqual(end, samples)
                    self.assertEqual(start % 800, 0)
                    self.assertEqual(end % 800, 0)
                    self.assertEqual((start // 160) * 160, (start // 400) * 400)
                    self.assertEqual((end // 160) * 160, (end // 400) * 400)
                    self.assertGreaterEqual(0.7 * samples - (end - start), 0)
                    self.assertLess(0.7 * samples - (end - start), 800)

    def test_mask_and_noise_seeds_do_not_depend_on_global_rng_state(self):
        key = "dev-clean/1/2/example-a"
        span = physical_span(127321, key)
        seeds = [keyed_seed(seed, "ode-noise", key) for seed in (666, 667, 668)]
        torch.manual_seed(31991)
        torch.randn(128)
        self.assertEqual(physical_span(127321, key), span)
        self.assertEqual([keyed_seed(seed, "ode-noise", key) for seed in (666, 667, 668)], seeds)
        self.assertEqual(len(set(seeds)), 3)
        self.assertNotEqual(keyed_seed(666, "context-posterior", key), seeds[0])


class WaveformIsolationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-pair-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        generator = torch.Generator().manual_seed(938)
        self.source = torch.randn((1, 16000), generator=generator) * 0.2
        source_path = self.root / "source.wav"
        save_wave(source_path, self.source)
        self.record = {
            "source_audio_path": str(source_path),
            "source_audio_sha256": sha256_file(source_path),
            "original_num_samples": 16000,
            "mask_start_sample": 4000,
            "mask_end_sample": 12000,
        }

    def bind_observed_waveform(self, value, name="observed.wav"):
        path = self.root / name
        save_wave(path, value)
        self.record["reference_input"] = artifact(path, Path(name))

    def test_accepts_shared_waveform_with_exactly_zero_hidden_region(self):
        observed = self.source.clone()
        observed[:, 4000:12000] = 0
        self.bind_observed_waveform(observed)
        loaded = load_observed_input(self.root, self.record)
        self.assertTrue(torch.equal(loaded, observed))
        self.assertEqual(torch.count_nonzero(loaded[:, 4000:12000]).item(), 0)
        self.assertTrue(torch.equal(loaded[:, :4000], self.source[:, :4000]))
        self.assertTrue(torch.equal(loaded[:, 12000:], self.source[:, 12000:]))

    def test_rejects_hidden_ground_truth_even_when_artifact_hash_is_valid(self):
        self.bind_observed_waveform(self.source)
        with self.assertRaisesRegex(RuntimeError, "exact zeros inside"):
            load_observed_input(self.root, self.record)

    def test_rejects_modified_observed_context_even_when_artifact_hash_is_valid(self):
        observed = self.source.clone()
        observed[:, 4000:12000] = 0
        observed[:, 20] += 0.125
        self.bind_observed_waveform(observed)
        with self.assertRaisesRegex(RuntimeError, "equal source outside mask"):
            load_observed_input(self.root, self.record)

    def test_rejects_source_changed_after_common_fixture_was_frozen(self):
        observed = self.source.clone()
        observed[:, 4000:12000] = 0
        self.bind_observed_waveform(observed)
        replacement = self.root / "replacement.wav"
        save_wave(replacement, self.source + 0.01)
        self.record["source_audio_path"] = str(replacement)
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            load_observed_input(self.root, self.record)


class ArtifactTests(unittest.TestCase):
    def test_float_wav_is_exact_without_clipping_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="native-pair-wave-") as temporary:
            path = Path(temporary) / "unclipped.wav"
            source = torch.tensor([[0.0, 1.5, -1.2, 0.4, 1e-8]], dtype=torch.float32)
            save_wave(path, source)
            loaded, sample_rate = torchaudio.load(str(path))
            self.assertEqual(sample_rate, 16000)
            self.assertTrue(torch.equal(loaded, source))
            original_hash = sha256_file(path)
            with self.assertRaises(ValueError):
                save_wave(path, torch.zeros_like(source))
            self.assertEqual(sha256_file(path), original_hash)

    def test_canary_output_is_separate_and_existing_final_output_is_refused(self):
        self.assertEqual(branch_name("mel", None), "mel")
        self.assertEqual(branch_name("mel", 2), "mel_canary2")
        with tempfile.TemporaryDirectory(prefix="native-pair-output-") as temporary:
            root = Path(temporary)
            (root / "mel").mkdir()
            with self.assertRaises(FileExistsError):
                create_attempt(root, "mel")
            attempt, final = create_attempt(root, "mel_canary2")
            self.assertTrue(attempt.is_dir())
            self.assertEqual(final, root / "mel_canary2")
            self.assertFalse(final.exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
