"""CPU smoke tests for the MingTok C2 data/model contract."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_mingtok import CFM_MingTok
from aligndit.model.dataset_mingtok import CELEBVDUB_RAW_ARROW_SHA256, CustomDatasetMingTokVideo
from aligndit.model.mingtok_codec import EXPECTED_CONFIG_SHA256, EXPECTED_MODEL_SHA256


def test_dataset_contract():
    with tempfile.TemporaryDirectory(prefix="mingtok_c2_smoke_") as temp_dir:
        root = Path(temp_dir)
        cache_dir = root / "cache"
        (cache_dir / "latents" / "train" / "speaker").mkdir(parents=True)
        (root / "data" / "CelebVDub" / "avhubert_video_feat" / "train" / "speaker").mkdir(parents=True)
        (root / "data" / "CelebVDub" / "audio" / "train" / "speaker").mkdir(parents=True)

        contract = {
            "schema_version": 1,
            "codec": "MingTok-Audio",
            "latent_dim": 64,
            "dtype": "float32",
            "layout": "T,D",
            "latent_fps": 50,
            "sample_rate": 16_000,
            "hop_size": 320,
            "audio_video_ratio": 2,
            "normalization": "none",
            "posterior_mode": "sample",
            "base_seed": 666,
            "split": "train",
            "num_items": 2,
            "selection": {
                "type": "raw_arrow",
                "metadata_sha256": CELEBVDUB_RAW_ARROW_SHA256,
                "audio_path_field": "audio_path",
                "ordering": "metadata_row_order",
            },
            "checkpoint": {
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "model_sha256": EXPECTED_MODEL_SHA256,
            },
        }
        with open(cache_dir / "contract.json", "w", encoding="utf-8") as file:
            json.dump(contract, file)

        audio_paths = []
        for clip, latent_len, video_len in (("pad", 7, 4), ("trim", 9, 3)):
            audio_path = root / "data" / "CelebVDub" / "audio" / "train" / "speaker" / f"{clip}.wav"
            audio_paths.append(str(audio_path))
            latent = np.arange(latent_len * 64, dtype=np.float32).reshape(latent_len, 64)
            video = np.arange(video_len * 8, dtype=np.float32).reshape(video_len, 8)
            np.save(cache_dir / "latents" / "train" / "speaker" / f"{clip}.npy", latent)
            np.save(
                root / "data" / "CelebVDub" / "avhubert_video_feat" / "train" / "speaker" / f"{clip}.npy",
                video,
            )

        metadata = Dataset.from_dict(
            {
                "audio_path": audio_paths,
                "text": ["hello", "test"],
                "duration": [0.16, 0.12],
            }
        )
        dataset = CustomDatasetMingTokVideo(
            metadata,
            data_dir=None,
            cache_dir=str(cache_dir),
            min_duration=0.0,
            video_dim=8,
        )
        padded = dataset.getitem(0)
        trimmed = dataset.getitem(1)
        assert padded["audio_latent"].shape == (64, 8)
        assert trimmed["audio_latent"].shape == (64, 6)
        torch.testing.assert_close(padded["audio_latent"][:, -1], padded["audio_latent"][:, -2])
        assert dataset.get_frame_len(0) == 8.0

        batch = dataset.collate_fn([padded, None, trimmed])
        assert batch["audio_latent"].shape == (2, 64, 8)
        assert batch["video"].shape == (2, 4, 8)
        assert batch["audio_latent_lengths"].tolist() == [8, 6]
        assert batch["video_lengths"].tolist() == [4, 3]
        assert batch["audio_latent"].dtype == torch.float32

        contract["posterior_mode"] = "mean"
        with open(cache_dir / "contract.json", "w", encoding="utf-8") as file:
            json.dump(contract, file)
        try:
            CustomDatasetMingTokVideo(
                metadata,
                data_dir=None,
                cache_dir=str(cache_dir),
                min_duration=0.0,
                video_dim=8,
            )
        except ValueError as error:
            assert "posterior_mode" in str(error)
        else:
            raise AssertionError("Dataset accepted a posterior-mean cache for the sampled-latent experiment")
        print("[OK] FP32 cache, legal-batch filtering, and exact 2:1 latent/video collation")


def build_tiny_model():
    transformer = DiT_VT_MMDiT(
        dim=32,
        depth=2,
        heads=4,
        dim_head=8,
        ff_mult=2,
        mel_dim=64,
        text_num_embeds=16,
        text_dim=16,
        text_mask_padding=False,
        qk_norm="rms_norm",
        conv_layers=0,
        pe_attn_head=1,
        attn_mask_enabled=True,
        checkpoint_activations=False,
        use_conformer=False,
        layer_indices_ctc=[0],
        ctc_sampling_ratios=[1, 1],
        n_mm_layers=1,
        n_text_layers=1,
        prompt_isolated_ca=False,
        audio_video_ratio=2,
        video_dim=8,
        video_rope_scaled=True,
    )
    model = CFM_MingTok(
        transformer,
        num_channels=64,
        audio_video_ratio=2,
        ctc_lambda=0.0,
    )
    return model


def test_train_and_ctc_shapes():
    torch.manual_seed(0)
    model = build_tiny_model().train()
    batch, audio_len, video_len, text_len = 2, 8, 4, 3
    latent = torch.randn(batch, audio_len, 64)
    video = torch.randn(batch, video_len, 8)
    text = torch.randint(0, 16, (batch, text_len))
    audio_lens = torch.tensor([8, 6])
    video_lens = torch.tensor([4, 3])
    text_lens = torch.tensor([3, 2])

    loss, components, _cond, pred = model(
        latent,
        text=text,
        video=video,
        lens=audio_lens,
        text_lens=text_lens,
        video_lens=video_lens,
    )
    assert pred.shape == latent.shape
    assert torch.isfinite(loss)
    assert set(components) == {"diff_loss"}
    loss.backward()

    transformer = model.transformer
    mask = torch.arange(audio_len).unsqueeze(0) < audio_lens.unsqueeze(1)
    video_mask = torch.arange(video_len).unsqueeze(0) < video_lens.unsqueeze(1)
    text_mask = torch.arange(text_len).unsqueeze(0) < text_lens.unsqueeze(1)
    with torch.no_grad():
        output, intermediates = transformer(
            x=latent,
            cond=torch.zeros_like(latent),
            text=text,
            video=video,
            time=torch.rand(batch),
            mask=mask,
            text_mask=text_mask,
            video_mask=video_mask,
            complementary_mask=torch.zeros_like(video_mask),
            generation_mask=mask,
        )
    assert output.shape == latent.shape
    assert intermediates[0]["z_lens"].tolist() == audio_lens.tolist()
    assert transformer.ctc_sampling_ratios == (1, 1)
    print("[OK] latent-only CFM forward/backward and 50 Hz CTC projector length")


def test_single_sample():
    torch.manual_seed(0)
    model = build_tiny_model().eval()
    cond = torch.randn(1, 4, 64)
    video = torch.randn(1, 4, 8)
    text = torch.randint(0, 16, (1, 3))
    with torch.no_grad():
        output, trajectory = model.sample(
            cond=cond,
            text=text,
            duration=8,
            video=video,
            lens=torch.tensor([4]),
            steps=1,
            cfg_strength=0.0,
            cfg_strength_v=0.0,
            use_epss=False,
        )
    assert output.shape == (1, 8, 64)
    assert trajectory.shape[1:] == output.shape
    torch.testing.assert_close(output[:, :4], cond)

    try:
        model.sample(cond=torch.randn(1, 640), text=text, duration=8, video=video)
    except ValueError as error:
        assert "[B, T, 64]" in str(error)
    else:
        raise AssertionError("CFM_MingTok accepted a waveform-like 2D condition")
    print("[OK] single-sample ODE path preserves the latent prompt and rejects waveform input")


def main():
    # Make CPU smoke behavior deterministic and avoid excessive thread startup.
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    test_dataset_contract()
    test_train_and_ctc_shapes()
    test_single_sample()
    print("All MingTok C2 smoke tests passed.")


if __name__ == "__main__":
    main()
