"""CPU contract tests for non-VAE C2 + explicit speaker conditioning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.cfm_vt import CFM_VT
from aligndit.model.speaker_embedding import (
    SpeakerEmbeddingError,
    load_speaker_embedding,
    speaker_embedding_path,
)


BASE_ARCH = {
    "dim": 32,
    "depth": 4,
    "heads": 4,
    "dim_head": 8,
    "ff_mult": 2,
    "mel_dim": 8,
    "text_num_embeds": 32,
    "text_dim": 16,
    "text_mask_padding": False,
    "qk_norm": "rms_norm",
    "conv_layers": 0,
    "pe_attn_head": 1,
    "attn_mask_enabled": True,
    "checkpoint_activations": False,
    "use_conformer": False,
    "layer_indices_ctc": [],
    "n_mm_layers": 2,
    "n_text_layers": 2,
    "prompt_isolated_ca": False,
    "audio_video_ratio": 4,
    "video_dim": 16,
    "video_rope_scaled": True,
}
SPEAKER_DIM = 6


def make_inputs(batch=2):
    audio_len = 8
    video_len = audio_len // 4
    return {
        "x": torch.randn(batch, audio_len, 8),
        "cond": torch.randn(batch, audio_len, 8),
        "text": torch.randint(0, 32, (batch, 5)),
        "video": torch.randn(batch, video_len, 16),
        "time": torch.tensor([0.2, 0.8])[:batch],
        "mask": torch.ones(batch, audio_len, dtype=torch.bool),
        "text_mask": torch.ones(batch, 5, dtype=torch.bool),
        "video_mask": torch.ones(batch, video_len, dtype=torch.bool),
        "complementary_mask": torch.ones(batch, video_len, dtype=torch.bool),
        "generation_mask": torch.tensor([[False, False, True, True, True, True, True, True]]).repeat(batch, 1),
        "cache": False,
    }


def test_zero_initialization_is_exact_c2():
    torch.manual_seed(7)
    base = DiT_VT_MMDiT(**BASE_ARCH).eval()
    torch.manual_seed(7)
    speaker_model = DiT_VT_MMDiT(
        **BASE_ARCH,
        speaker_dim=SPEAKER_DIM,
        speaker_condition_start_layer=2,
    ).eval()

    base_state = base.state_dict()
    speaker_state = speaker_model.state_dict()
    assert set(speaker_state) - set(base_state) == {"speaker_proj.weight"}
    assert all(torch.equal(value, speaker_state[key]) for key, value in base_state.items())
    assert torch.count_nonzero(speaker_model.speaker_proj.weight) == 0
    assert speaker_model.speaker_proj.bias is None

    inputs = make_inputs()
    speaker_embedding = torch.randn(2, SPEAKER_DIM)
    with torch.inference_mode():
        base_output, _ = base(**inputs)
        speaker_output, _ = speaker_model(**inputs, speaker_embedding=speaker_embedding)
    assert torch.equal(base_output, speaker_output)
    print("[OK] zero-initialized speaker path is exactly equivalent to C2")


def test_tail_only_cfg_branch_order():
    torch.manual_seed(11)
    model = DiT_VT_MMDiT(
        **BASE_ARCH,
        speaker_dim=SPEAKER_DIM,
        speaker_condition_start_layer=2,
    ).eval()
    with torch.no_grad():
        model.speaker_proj.weight.copy_(
            torch.arange(model.dim * SPEAKER_DIM, dtype=torch.float32).reshape(model.dim, SPEAKER_DIM) / 1000
        )

    inputs = make_inputs()
    speaker_embedding = torch.randn(2, SPEAKER_DIM)
    captured_t = {}

    def capture_mm(layer_i):
        def hook(_module, args):
            captured_t[layer_i] = args[2].detach().clone()

        return hook

    def capture_audio(layer_i):
        def hook(_module, args):
            captured_t[layer_i] = args[1].detach().clone()

        return hook

    hooks = [
        model.transformer_blocks[0].register_forward_pre_hook(capture_mm(0)),
        model.transformer_blocks[2].register_forward_pre_hook(capture_audio(2)),
    ]
    try:
        with torch.inference_mode():
            model(**inputs, speaker_embedding=speaker_embedding, cfg_infer=True)
    finally:
        for hook in hooks:
            hook.remove()

    base_t = model.time_embed(inputs["time"])
    speaker_delta = model.get_speaker_delta(speaker_embedding, base_t)
    expected_front_t = base_t.repeat((3, 1))
    expected_tail_t = torch.cat(
        [base_t + speaker_delta, base_t + speaker_delta, base_t],
        dim=0,
    )
    assert torch.allclose(captured_t[0], expected_front_t)
    assert torch.allclose(captured_t[2], expected_tail_t)
    assert torch.count_nonzero(model.get_speaker_delta(speaker_embedding, base_t, drop_speaker=True)) == 0
    print("[OK] speaker delta reaches only the audio tail; B=2 CFG is full/TTS/null branch-major")


def test_cache_contract():
    with tempfile.TemporaryDirectory() as temporary_dir:
        root = Path(temporary_dir)
        audio = root / "CelebVDub/audio/train/video/clip.wav"
        cache = root / "CelebVDub/campplus_spk_emb_zh_en_16k"
        cache_path = speaker_embedding_path(audio, cache)
        cache_path.parent.mkdir(parents=True)
        embedding = np.arange(1, 193, dtype=np.float32)
        embedding /= np.linalg.norm(embedding)
        np.save(cache_path, embedding, allow_pickle=False)
        loaded = load_speaker_embedding(audio, cache)
        assert loaded.shape == (192,) and loaded.dtype == torch.float32
        assert torch.allclose(loaded, torch.from_numpy(embedding))

        np.save(cache_path, np.zeros(192, dtype=np.float32), allow_pickle=False)
        try:
            load_speaker_embedding(audio, cache)
        except SpeakerEmbeddingError:
            pass
        else:
            raise AssertionError("zero-norm cache was accepted")
    print("[OK] mirrored cache loader fails fast on invalid speaker embeddings")


def test_batched_cfm_sampling_contract():
    torch.manual_seed(13)
    transformer = DiT_VT_MMDiT(
        **BASE_ARCH,
        speaker_dim=SPEAKER_DIM,
        speaker_condition_start_layer=2,
    )
    with torch.no_grad():
        transformer.speaker_proj.weight.normal_(std=0.01)
    model = CFM_VT(transformer=transformer, num_channels=8, ctc_lambda=0.0)
    generated, _ = model.sample(
        cond=torch.randn(2, 4, 8),
        text=torch.randint(0, 32, (2, 5)),
        duration=torch.tensor([9, 8]),
        video=torch.randn(2, 4, 16),
        speaker_embedding=torch.randn(2, SPEAKER_DIM),
        lens=torch.tensor([4, 4]),
        steps=1,
        cfg_strength=1.0,
        cfg_strength_v=1.0,
        use_epss=False,
        seed=0,
    )
    assert generated.shape == (2, 9, 8)
    assert torch.isfinite(generated).all()
    print("[OK] B=2 CFM sampling handles non-multiple-of-four duration/video masks")


def main():
    test_zero_initialization_is_exact_c2()
    test_tail_only_cfg_branch_order()
    test_cache_contract()
    test_batched_cfm_sampling_contract()
    print("All C2 speaker-conditioning CPU smoke tests passed.")


if __name__ == "__main__":
    main()
