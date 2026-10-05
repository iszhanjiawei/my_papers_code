"""CPU regressions for frozen external-reference selection and cache safety."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from aligndit.script.eval.prepare_grid_reference import (
    EXPECTED_SPEAKERS,
    atomic_bytes,
    extract_reference,
    read_grid_transcript,
    select_references,
    stable_reference_seed,
    validate_cached_record,
)
from aligndit.script.eval.semantic_vae_decoder import load_semantic_vae_decoder


def inventory():
    return {s: [{"speaker": s, "clip": f"clip{i:04d}"} for i in range(20)] for s in EXPECTED_SPEAKERS}


def check_balanced_unique_deterministic_selection():
    refs = select_references(inventory(), 213, 0)
    assert len({(item["speaker"], item["clip"]) for item in refs}) == 213
    assert {sum(item["speaker"] == s for item in refs) for s in EXPECTED_SPEAKERS} == {6, 7}
    assert "s21" not in {item["speaker"] for item in refs}
    reordered = {s: list(reversed(items)) for s, items in reversed(list(inventory().items()))}
    assert refs == select_references(reordered, 213, 0)
    assert refs != select_references(inventory(), 213, 1)


def check_selection_rejects_insufficient_unique_references():
    with unittest.TestCase().assertRaisesRegex(ValueError, "Insufficient"):
        select_references({"s1": [{"clip": "a"}]}, 2, 0)


def check_stable_seed_independent_of_pair_order():
    ids = ["grid/s1/bbaf2n", "grid/s2/bbaf2n"]
    expected = {key: stable_reference_seed(key) for key in ids}
    assert {key: stable_reference_seed(key) for key in reversed(ids)} == expected
    assert len(set(expected.values())) == 2
    assert all(0 <= value < 2**63 for value in expected.values())
    with unittest.TestCase().assertRaises(ValueError):
        stable_reference_seed("grid/s1/../escape")


def check_transcript_is_full_text_not_alignment(tmp_path):
    lab = tmp_path / "reference.lab"
    lab.write_text(" BIN blue at f two now\n", encoding="utf-8")
    assert read_grid_transcript(lab) == "bin blue at f two now"
    lab.write_text("lay sp red sp by i eight please", encoding="utf-8")
    assert read_grid_transcript(lab) == "lay red by i eight please"
    lab.write_text("0 1234 sil\n1234 9999 bin\n", encoding="utf-8")
    with unittest.TestCase().assertRaisesRegex(ValueError, "six-word"):
        read_grid_transcript(lab)


def check_immutable_publication_refuses_repairing(tmp_path):
    path = tmp_path / "pairs.jsonl"
    atomic_bytes(path, b"original\n")
    atomic_bytes(path, b"original\n")
    with unittest.TestCase().assertRaisesRegex(FileExistsError, "differs"):
        atomic_bytes(path, b"different\n")
    assert path.read_bytes() == b"original\n"


class FakeVae:
    @staticmethod
    def encoder(waveform):
        return waveform.reshape(1, 1, -1, 400).mean(dim=-1).expand(-1, 64, -1)

    @staticmethod
    def pre_block(hidden):
        return hidden

    @staticmethod
    def fc_mu(hidden):
        return hidden

    @staticmethod
    def fc_var(hidden):
        return torch.zeros_like(hidden)


class FakeCampplus:
    def __call__(self, features):
        return torch.arange(1, 193, dtype=torch.float32).unsqueeze(0).expand(features.shape[0], -1)


def check_reference_features_share_exact_canonical_waveform_and_detect_tampering(tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.sin(np.arange(10001, dtype=np.float32) * 0.1) * 0.1, 25000, subtype="PCM_16")
    row = {
        "ref_id": "grid/s1/bbaf2n",
        "ref_source_audio": str(source),
        "ref_audio": str(tmp_path / "canonical.wav"),
        "ref_num_samples": 6401,
        "ref_frames": 17,
        "ref_latent_path": str(tmp_path / "latent.npy"),
        "ref_speaker_path": str(tmp_path / "speaker.npy"),
        "posterior_seed": stable_reference_seed("grid/s1/bbaf2n"),
    }
    record = extract_reference(row, FakeVae(), FakeCampplus(), torch.device("cpu"))
    validate_cached_record(record, row)
    audio, rate = sf.read(row["ref_audio"])
    assert rate == 16000 and audio.shape == (6401,)
    assert sf.info(row["ref_audio"]).subtype == "PCM_16"
    assert np.load(row["ref_latent_path"]).shape == (17, 64)
    assert np.isclose(np.linalg.norm(np.load(row["ref_speaker_path"])), 1)
    with unittest.TestCase().assertRaisesRegex(FileExistsError, "Untracked waveform"):
        extract_reference(row, FakeVae(), FakeCampplus(), torch.device("cpu"))
    Path(row["ref_audio"]).write_bytes(b"tampered")
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "hash mismatch"):
        validate_cached_record(record, row)


def check_cache_cannot_be_reused_with_different_pair():
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "plan mismatch"):
        validate_cached_record({"ref_id": "grid/s1/one"}, {"ref_id": "grid/s2/two"})


def check_decoder_compatibility_retains_cpu_loading_then_selected_device():
    calls = []
    model = torch.nn.Module()
    model.decoder = torch.nn.Linear(2, 2).eval().requires_grad_(False)
    metadata = {"checkpoint_sha256": hashlib.sha256(b"checkpoint").hexdigest()}

    def fake_loader(**kwargs):
        calls.append(kwargs)
        return model, metadata

    with patch("aligndit.script.eval.semantic_vae_decoder.load_semantic_vae", fake_loader):
        decoder, returned = load_semantic_vae_decoder(
            repo=Path("repo"), checkpoint_root=Path("checkpoint"), cache_spec={}, device=torch.device("cpu")
        )
    assert decoder is model.decoder and returned is metadata
    assert calls[0]["device"] == torch.device("cpu")
    assert not decoder.training and not decoder.weight.requires_grad


class GridReferencePreparationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="grid-reference-test-")
        self.addCleanup(temporary.cleanup)
        self.tmp_path = Path(temporary.name)

    def test_balanced_unique_deterministic_selection(self):
        check_balanced_unique_deterministic_selection()

    def test_selection_rejects_insufficient_unique_references(self):
        check_selection_rejects_insufficient_unique_references()

    def test_stable_seed_independent_of_pair_order(self):
        check_stable_seed_independent_of_pair_order()

    def test_transcript_is_full_text_not_alignment(self):
        check_transcript_is_full_text_not_alignment(self.tmp_path)

    def test_immutable_publication_refuses_repairing(self):
        check_immutable_publication_refuses_repairing(self.tmp_path)

    def test_reference_features_share_exact_waveform(self):
        check_reference_features_share_exact_canonical_waveform_and_detect_tampering(self.tmp_path)

    def test_cache_cannot_be_reused_with_different_pair(self):
        check_cache_cannot_be_reused_with_different_pair()

    def test_decoder_compatibility(self):
        check_decoder_compatibility_retains_cpu_loading_then_selected_device()


if __name__ == "__main__":
    unittest.main()
