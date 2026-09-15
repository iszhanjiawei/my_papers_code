#!/usr/bin/env python3
"""CPU contracts: test-only encoder and generated videos; no production caches."""
import tempfile
from pathlib import Path
import unittest
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch
import extract_synchformer as extraction
import av
import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
from aligndit.model.synchformer_features import (
    CHECKPOINT_SHA256, MODEL_ID, PREPROCESSING, SCHEMA_VERSION, FrozenSynchformerExtractor,
    atomic_json, cache_path, canonical_clip_key, decoded_video_frames, load_synchformer_feature,
    load_synchformer_payload, save_synchformer_feature, validate_synchformer_cache,
)

class FixtureExtractor(FrozenSynchformerExtractor):
    """Real decode/transform/windowing with a test-only encoder, requiring no weights."""
    def __init__(self):
        nn.Module.__init__(self)
        self.device = torch.device("cpu")
        self.batch_size = 8
        self.checkpoint_sha256 = CHECKPOINT_SHA256
        self.preprocess = v2.Compose([
            v2.Resize(224, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(224), v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5] * 3, [0.5] * 3)])
    def forward(self, segments):
        return segments[:, ::2].mean((2, 3, 4)).unsqueeze(-1).expand(-1, -1, 768).contiguous()

def make_video(path, count):
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as output:
        stream = output.add_stream("libx264", rate=25)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index % 200, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame): output.mux(packet)
        for packet in stream.encode(): output.mux(packet)

class CacheContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temporary = tempfile.TemporaryDirectory(prefix="synchformer_test_only_")
        cls.root = Path(cls.temporary.name)
        cls.cache = cls.root / "cache"
        cls.extractor = FixtureExtractor()
        cls.short_key, cls.long_key = "train/video_a/clip", "train/video_b/clip"
        cls.payloads = {}
        for key, count in [(cls.short_key, 11), (cls.long_key, 401)]:
            path = cls.root / (key + ".mp4")
            make_video(path, count)
            cls.payloads[key] = cls.extractor.extract(path, key)
            save_synchformer_feature(cls.cache, key, cls.payloads[key])
    @classmethod
    def tearDownClass(cls): cls.temporary.cleanup()
    def test_short_padding_and_long_clip_without_duration_cap(self):
        short = load_synchformer_payload(self.cache, self.short_key)
        self.assertEqual(short["features"].shape, (8, 768))
        self.assertEqual(short["metadata"]["padded_frames"], 5)
        self.assertEqual(short["metadata"]["num_sampled_frames"], 11)
        long = load_synchformer_payload(self.cache, self.long_key)
        self.assertEqual(long["metadata"]["num_sampled_frames"], 401)
        self.assertEqual(long["metadata"]["duration_seconds"], 16.04)
        self.assertEqual(long["features"].shape, (392, 768))
        self.assertEqual(long["metadata"]["unwindowed_tail_frames"], 1)
    def test_full_relative_keys_and_checkpoint_identity(self):
        self.assertNotEqual(cache_path(self.cache, self.short_key), cache_path(self.cache, self.long_key))
        self.assertEqual(canonical_clip_key("celebvdub/train/video_a/clip"), self.short_key)
        with self.assertRaises(ValueError): canonical_clip_key("../clip")
        with self.assertRaises(ValueError):
            load_synchformer_feature(self.cache, self.short_key, expected_checkpoint_sha256="wrong")
    def test_changed_source_and_nonfinite_features_rejected(self):
        payload = torch.load(cache_path(self.cache, self.short_key), weights_only=True)
        payload["features"][0, 0] = float("nan")
        save_synchformer_feature(self.cache, self.short_key, payload)
        with self.assertRaises(ValueError): load_synchformer_feature(self.cache, self.short_key)
        save_synchformer_feature(self.cache, self.short_key, self.payloads[self.short_key])
        with self.assertRaises(ValueError):
            load_synchformer_payload(self.cache, self.short_key, video_path=self.root / (self.long_key + ".mp4"))
    def test_full_audit_count_inventory_and_key_requirements(self):
        report = {"complete": True, "schema_version": SCHEMA_VERSION, "model_id": MODEL_ID,
            "checkpoint_sha256": CHECKPOINT_SHA256, "preprocessing": PREPROCESSING,
            "valid_keys": [self.short_key, self.long_key], "valid": 2, "expected_count": 2,
            "invalid": 0, "missing": 0, "inventory_sha256": "fixture"}
        path = self.cache / "coverage_report.json"
        atomic_json(path, report)
        validate_synchformer_cache(self.cache, expected_keys=[self.short_key], expected_manifest_sha256="fixture")
        with self.assertRaises(ValueError): validate_synchformer_cache(self.cache, expected_manifest_sha256="wrong")
        with self.assertRaises(ValueError): validate_synchformer_cache(self.cache, expected_keys=["test/video_a/clip"])
        report["expected_count"] = 3
        atomic_json(path, report)
        with self.assertRaises(ValueError): validate_synchformer_cache(self.cache)

class DecoderLifecycle(unittest.TestCase):
    def test_codec_closes_after_early_stop_and_decode_exception(self):
        with tempfile.TemporaryDirectory(prefix="synchformer_decoder_test_") as directory:
            video = Path(directory)/"clip.mp4"
            make_video(video, 11)
            original_open = av.open
            codecs = []
            def capture(*args, **kwargs):
                container = original_open(*args, **kwargs)
                codecs.append(container.streams.video[0].codec_context)
                return container
            with patch.object(av, "open", side_effect=capture):
                with decoded_video_frames(video) as frames:
                    next(frames)
                    self.assertTrue(codecs[-1].is_open)
                self.assertFalse(codecs[-1].is_open)
                with self.assertRaisesRegex(RuntimeError, "test decode failure"):
                    with decoded_video_frames(video) as frames:
                        next(frames)
                        raise RuntimeError("test decode failure")
                self.assertFalse(codecs[-1].is_open)

    def test_host_cleanup_runs_after_inner_scope_and_on_errors(self):
        extractor = FixtureExtractor()
        extractor.cleanup_interval = 2
        with patch.object(extractor, "_extract_video", return_value={"test":True}), \
             patch("aligndit.model.synchformer_features.release_extraction_host_memory") as cleanup:
            extractor.extract("unused")
            cleanup.assert_not_called()
            extractor.extract("unused")
            cleanup.assert_called_once()
        with patch.object(extractor, "_extract_video", side_effect=RuntimeError("fixture")), \
             patch("aligndit.model.synchformer_features.release_extraction_host_memory") as cleanup:
            with self.assertRaises(RuntimeError):
                extractor.extract("unused")
            cleanup.assert_called_once()
            self.assertEqual(extractor._clips_since_cleanup, 0)

class CacheStartupRace(unittest.TestCase):
    def test_transient_missing_or_incomplete_metadata_is_retried(self):
        expected = {"schema_version": 1}
        failures = [FileNotFoundError("temporarily invisible"), json.JSONDecodeError("transient", "", 0), json.dumps(expected)]
        with patch.object(Path, "read_text", side_effect=failures) as reader:
            observed = extraction.read_cache_metadata(Path("metadata.json"), attempts=3, retry_seconds=0)
        self.assertEqual(observed, expected)
        self.assertEqual(reader.call_count, 3)

    def test_parent_initializes_then_concurrent_workers_do_not_write(self):
        with tempfile.TemporaryDirectory(prefix="synchformer_startup_test_") as directory:
            root = Path(directory)
            inventory = root / "inventory.jsonl"
            inventory.write_text('{}\n')
            args = SimpleNamespace(cache_dir=root / "cache", inventory=inventory,
                checkpoint_sha256=CHECKPOINT_SHA256, video_root=root / "video")
            records = [{}]
            with patch.object(extraction.time, "sleep"):
                extraction.prepare_cache_metadata(args, records, initialize=True)
            metadata_path = args.cache_dir / "metadata.json"
            before_bytes, before_stat = metadata_path.read_bytes(), metadata_path.stat()
            # Matching parent initialization must preserve metadata byte-for-byte.
            extraction.prepare_cache_metadata(args, records, initialize=True)
            self.assertEqual(metadata_path.read_bytes(), before_bytes)
            self.assertEqual(metadata_path.stat().st_mtime_ns, before_stat.st_mtime_ns)
            coverage_path = args.cache_dir / "coverage_report.json"
            coverage_path.write_text("existing audit certificate")
            with patch.object(extraction, "atomic_json", side_effect=AssertionError("worker attempted metadata mutation")):
                with ThreadPoolExecutor(max_workers=16) as pool:
                    results = list(pool.map(lambda _: extraction.prepare_cache_metadata(args, records), range(16)))
            self.assertTrue(all(result["expected_count"] == 1 for result in results))
            self.assertEqual(metadata_path.read_bytes(), before_bytes)
            self.assertEqual(coverage_path.read_text(), "existing audit certificate")

if __name__ == "__main__": unittest.main()
