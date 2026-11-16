"""CPU publication tests against same-shape source swaps, duplicates and locks."""

from __future__ import annotations

import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import lrs3_finalize_cache as finalizer
from lrs3_build_inventory import digest, write_json, write_jsonl


class PublicationIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        stem = "train/pretrain/example_00001"
        self.row = {
            "sample_id": "example_00001", "utterance_key": f"lrs3/{stem}", "relative_stem": stem,
            "original_clip_id": "pretrain/example/00001", "source_split": "pretrain", "split": "train",
            "text": "hello", "text_status": "original", "duration_eligible": True,
            "audio_seconds": 1.0, "video_seconds": 1.0, "video_frames": 25, "latent_frames": 40,
            "posterior_seed": 666, "audio_path": str(self.root / "source.wav"),
            "video_path": str(self.root / "source.mp4"), "audio_relative_path": f"audio/{stem}.wav",
            "mouth_relative_path": f"video_mouth/{stem}.mp4", "label_source": "fixture",
            "label_validation": "fixture",
        }
        for name, shape, directory in (("latent", (40, 64), "latents"),
                                       ("video_25hz", (25, 1024), "avhubert_video_25hz"),
                                       ("video_40hz", (40, 1024), "video_40hz"),
                                       ("speaker", (192,), "speaker_embeddings")):
            relative = f"{directory}/{stem}.npy"
            self.row[f"{name}_relative_path"] = relative
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            array = np.ones(shape, dtype=np.float32)
            if name == "speaker":
                array /= np.linalg.norm(array)
            np.save(path, array)
        Path(self.row["audio_path"]).write_bytes(b"test audio source A")
        Path(self.row["video_path"]).write_bytes(b"test video source A")
        audio = self.root / self.row["audio_relative_path"]
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.symlink_to(self.row["audio_path"])
        mouth = self.root / self.row["mouth_relative_path"]
        mouth.parent.mkdir(parents=True, exist_ok=True)
        mouth.write_bytes(b"test mouth video")
        hashes = {name: digest(self.root / self.row[f"{name}_relative_path"])
                  for name in ("latent", "video_25hz", "video_40hz", "speaker")}
        write_json(self.root / f"reports/media_receipts/{stem}.json", {
            "source_path": self.row["video_path"], "source_sha256": digest(self.row["video_path"]),
            "feature_sha256": hashes["video_25hz"], "features_complete": True, "video_frames": 25,
            "protocol_sha256": "29d918bfbcb10e9bca98b1ff283c273618231f57159983d236d5f4b367df720f",
            "crop_kind": "fixture_mouth",
        })
        write_json(self.root / f"state/audio_receipts/{stem}.json", {
            "identity": {
                "audio_sha256": digest(self.row["audio_path"]), "utterance_key": self.row["utterance_key"],
                "posterior_seed": 666, "posterior_protocol": "semantic_vae_posterior_sample_v1",
                "codec_sha256": "7c455aa8ab3f7d576b4834f8342558894aafaa61a371b84a9bfa4d10a100e516",
                "campplus_sha256": "92f29b94e6948786a26778c9e302525d185bb08c8b9f5252ed98776902840199",
                "speaker_protocol": "full_waveform_10s_repeat_fbank_mean_average_l2_v1",
            },
            "outputs": {key: hashes[key] for key in ("latent", "speaker")},
        })
        write_json(self.root / f"state/video40_receipts/{stem}.json", {
            "identity": {"source_sha256": hashes["video_25hz"],
                         "protocol": "avhubert_video_25hz_to_40hz_linear_align_corners_false_v1",
                         "utterance_key": self.row["utterance_key"], "target_frames": 40},
            "outputs": {"video_40hz": hashes["video_40hz"]},
        })

    def validate(self):
        return finalizer.validate(self.row, self.root, {c: i for i, c in enumerate(" helo")}, {}, True)

    def run_publication(self, rows):
        source = self.root / "source.jsonl"
        write_jsonl(source, rows)
        return subprocess.run([sys.executable, str(Path(finalizer.__file__)), "--root", str(self.root),
                               "--source-manifest", str(source), "--name", "test_publication", "--mode", "full"],
                              capture_output=True, text=True, timeout=30)

    def test_valid_source_bound_receipts_pass(self):
        checked, reasons = self.validate()
        self.assertEqual(reasons, [])
        self.assertEqual(checked["source_audio_sha256"], digest(self.row["audio_path"]))

    def test_same_shape_latent_swap_rejected(self):
        np.save(self.root / self.row["latent_relative_path"], np.full((40, 64), 2.0, dtype=np.float32))
        _, reasons = self.validate()
        self.assertTrue(any(reason.startswith("audio_provenance:") for reason in reasons))

    def test_changed_source_audio_rejected(self):
        Path(self.row["audio_path"]).write_bytes(b"test audio source B")
        _, reasons = self.validate()
        self.assertTrue(any(reason.startswith("audio_provenance:") for reason in reasons))

    def test_changed_source_video_rejected(self):
        Path(self.row["video_path"]).write_bytes(b"test video source B")
        _, reasons = self.validate()
        self.assertTrue(any(reason.startswith("media_provenance:") for reason in reasons))

    def test_same_shape_video40_swap_rejected(self):
        np.save(self.root / self.row["video_40hz_relative_path"], np.full((40, 1024), 2.0, dtype=np.float32))
        _, reasons = self.validate()
        self.assertTrue(any(reason.startswith("video40_provenance:") for reason in reasons))

    def test_wrong_posterior_seed_rejected(self):
        self.row["posterior_seed"] += 1
        _, reasons = self.validate()
        self.assertTrue(any(reason.startswith("audio_provenance:") for reason in reasons))

    def recovery_label(self, pseudo=False):
        label = {'sample_id': self.row['sample_id'], 'original_clip_id': self.row['original_clip_id'],
                 'text': 'hello hello hello', 'accepted': True, 'source': 'test evidence', 'quality': 'test',
                 'source_kind': 'asr_pseudo' if pseudo else 'source_asr_consensus',
                 'source_audio_sha256': digest(self.row['audio_path']),
                 'independent_content_verified': False, 'acceptance_policy_version': 'lrs3_asr_pseudo_p1_v1',
                 'confidence': {key: threshold for key, (_, threshold) in finalizer.PSEUDO_THRESHOLDS.items()},
                 'quality_gates': {key: True for key in ['ctc_text_exact', 'group_contiguous_support',
                                    'word_text_exact', 'positive_word_durations', 'within_pcm_bounds',
                                    'group_boundary_consistent']},
                 'evidence_refs': [{'path': self.row['audio_path'], 'sha256': digest(self.row['audio_path'])}]}
        self.row['text'] = ''
        return label

    def validate_label(self, label, policy='source_grounded'):
        return finalizer.validate(self.row, self.root, {c: i for i, c in enumerate(' helo')},
                                  {self.row['sample_id']: label}, True, policy)

    def test_recovered_text_must_bind_current_source_audio(self):
        label = self.recovery_label()
        label['source_audio_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'actual source audio'):
            self.validate_label(label)

    def test_strict_policy_rejects_pseudo_even_with_all_checks_passed(self):
        label = self.recovery_label(pseudo=True)
        with self.assertRaisesRegex(ValueError, 'explicit training-only extension'):
            self.validate_label(label)

    def test_pseudo_policy_preserves_explicit_provenance(self):
        label = self.recovery_label(pseudo=True)
        checked, reasons = self.validate_label(label, 'source_plus_asr_pseudo')
        self.assertEqual(reasons, [])
        self.assertEqual(checked['text_status'], 'asr_pseudo')
        self.assertEqual(checked['source_kind'], 'asr_pseudo')
        self.assertFalse(checked['independent_content_verified'])
        self.assertEqual(checked['confidence'], label['confidence'])

    def test_pseudo_cannot_enter_test_split(self):
        label = self.recovery_label(pseudo=True)
        self.row['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'explicit training-only extension'):
            self.validate_label(label, 'source_plus_asr_pseudo')

    def test_pseudo_bad_confidence_and_missing_boundary_check_rejected(self):
        label = self.recovery_label(pseudo=True)
        label['confidence']['ctc_min_char_probability'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.validate_label(label, 'source_plus_asr_pseudo')
        label['confidence']['ctc_min_char_probability'] = 0.9
        label['quality_gates']['within_pcm_bounds'] = False
        with self.assertRaisesRegex(ValueError, 'boundary checks'):
            self.validate_label(label, 'source_plus_asr_pseudo')

    def run_extension(self):
        return subprocess.run([sys.executable, str(Path(finalizer.__file__)), '--root', str(self.root),
                               '--source-manifest', str(self.root / 'source.jsonl'), '--name', 'extension',
                               '--mode', 'full', '--text-policy', 'source_plus_asr_pseudo',
                               '--strict-baseline', 'manifests/test_publication'],
                              capture_output=True, text=True, timeout=30)

    def test_extension_pins_baseline_and_preserves_evaluation_bytes(self):
        strict = self.run_publication([self.row])
        self.assertEqual(strict.returncode, 0, strict.stderr)
        extension = self.run_extension()
        self.assertEqual(extension.returncode, 0, extension.stderr)
        old = self.root / 'manifests/test_publication'
        new = self.root / 'manifests/extension'
        for name in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
            self.assertEqual((old / name).read_bytes(), (new / name).read_bytes())
        policy = json.loads((new / 'text_policy.json').read_text())
        self.assertEqual(policy['strict_baseline']['inventory_meta_sha256'], digest(old / 'inventory_meta.json'))
        receipt = json.loads((new / 'cache_completion.json').read_text())
        self.assertEqual(receipt['text_policy_sha256'], digest(new / 'text_policy.json'))

    def test_extension_cannot_modify_existing_strict_transcript(self):
        strict = self.run_publication([self.row])
        self.assertEqual(strict.returncode, 0, strict.stderr)
        changed = dict(self.row, text='ello')
        write_jsonl(self.root / 'source.jsonl', [changed])
        extension = self.run_extension()
        self.assertNotEqual(extension.returncode, 0)
        self.assertIn('changed or removed a strict baseline record', extension.stderr)
        self.assertFalse((self.root / 'manifests/extension/cache_completion.json').exists())

    def test_duplicate_source_identity_rejected_before_publication(self):
        result = self.run_publication([self.row, self.row])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Duplicate source sample_id", result.stderr)
        self.assertFalse((self.root / "manifests/test_publication/cache_completion.json").exists())

    def test_test_source_video_cannot_cross_training_split(self):
        other = dict(self.row)
        original = self.row['original_clip_id'].split('/')
        other.update(sample_id='heldout_other_clip', utterance_key='lrs3/test/test/heldout_other_clip',
                     relative_stem='test/test/heldout_other_clip', split='test',
                     original_clip_id=f'test/{original[1]}/99999')
        result = self.run_publication([self.row, other])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('A test source video also appears', result.stderr)
        self.assertFalse((self.root / 'manifests/test_publication/cache_completion.json').exists())

    def test_active_writer_lock_rejects_publication(self):
        lock_path = self.root / "state/locks/audio.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = self.run_publication([self.row])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BlockingIOError", result.stderr)
        self.assertFalse((self.root / "manifests/test_publication/cache_completion.json").exists())

    def test_unfinished_writer_guard_rejects_publication(self):
        write_json(self.root / "state/video_25hz/WRITE_ACTIVE.json", {"pid": 123})
        result = self.run_publication([self.row])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cache writes are unfinished: video_25hz", result.stderr)
        self.assertFalse((self.root / "manifests/test_publication/cache_completion.json").exists())


if __name__ == "__main__":
    unittest.main()
