"""CPU Synchformer contracts. Run with PYTHONPATH=src; no data/weights needed."""
from __future__ import annotations
from unittest.mock import patch
import torch
import torch.nn.functional as F
from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT, _resample_valid_sync
from aligndit.model.cfm_vt import CFM_VT
from smoke_test_semantic_vae_c2_speaker import BASE_ARCH, SPEAKER_DIM, make_inputs

SYNC_DIM = 768


def make_pair(checkpoint=False, audio_video_ratio=1):
    arch = {**BASE_ARCH, "n_mm_layers": 1, "n_text_layers": 2,
            "speaker_dim": SPEAKER_DIM, "speaker_condition_start_layer": 2,
            "checkpoint_activations": checkpoint, "dropout": 0.0, "audio_video_ratio": audio_video_ratio}
    torch.manual_seed(17)
    parent = DiT_VT_MMDiT(**arch)
    with torch.no_grad():
        for block in parent.transformer_blocks:
            block.attn_norm.linear.weight.normal_(std=0.03)
            block.attn_norm.linear.bias.normal_(std=0.03)
        parent.transformer_blocks[0].v_attn_norm.linear.weight.normal_(std=0.03)
        parent.transformer_blocks[0].cross_attn_ada.weight.normal_(std=0.03)
        parent.proj_out.weight.normal_(std=0.03)
        parent.norm_out.linear.weight.normal_(std=0.03)
        parent.speaker_proj.weight.normal_(std=0.03)
    model = DiT_VT_MMDiT(**arch, sync_dim=SYNC_DIM)
    missing, unexpected = model.load_state_dict(parent.state_dict(), strict=False)
    expected = {"sync_pos_emb", "sync_in.0.weight", "sync_in.0.bias", "sync_in.2.w1.weight",
                "sync_in.2.w2.weight", "sync_in.2.w3.weight"}
    assert set(missing) == expected and not unexpected
    return parent, model


def inputs():
    return {**make_inputs(), "speaker_embedding": torch.randn(2, SPEAKER_DIM),
            "sync_feat": torch.randn(2, 24, SYNC_DIM), "sync_lens": torch.tensor([24, 16])}


def sync_deltas(model, data, **flags):
    return model.get_sync_deltas(
        data["sync_feat"], data["sync_lens"], model.time_embed(data["time"]),
        audio_len=data["x"].shape[1], video_len=data["video"].shape[1], audio_mask=data["mask"], video_mask=data["video_mask"],
        complementary_mask=data["complementary_mask"], **flags)


def test_valid_interpolation():
    source = torch.randn(3, 32, 7)
    source_lens, target_lens = torch.tensor([32, 16, 8]), torch.tensor([53, 23, 2])
    actual = _resample_valid_sync(source, source_lens, target_lens, 53)
    for row, (src_len, tgt_len) in enumerate(zip(source_lens, target_lens)):
        expected = F.interpolate(source[row, :src_len].T[None], size=int(tgt_len), mode="nearest-exact")[0].T
        torch.testing.assert_close(actual[row, :tgt_len], expected, rtol=0, atol=0)
        assert not torch.count_nonzero(actual[row, tgt_len:])
    print("[OK] mixed-length interpolation matches independent nearest-exact clips")



def test_same_seed_initialization_and_rates():
    arch = {**BASE_ARCH, "speaker_dim": SPEAKER_DIM, "speaker_condition_start_layer": 2}
    torch.manual_seed(666)
    baseline = DiT_VT_MMDiT(**arch)
    torch.manual_seed(666)
    added = DiT_VT_MMDiT(**arch, sync_dim=SYNC_DIM)
    for key, value in baseline.state_dict().items():
        torch.testing.assert_close(value, added.state_dict()[key], rtol=0, atol=0)
    _, model = make_pair(audio_video_ratio=2)
    model.eval()
    data = inputs()
    data["video"] = data["video"][:, :6]
    data["video_mask"] = torch.arange(6)[None] < torch.tensor([6, 5])[:, None]
    data["complementary_mask"] = torch.zeros(2, 6, dtype=torch.bool)
    data["complementary_mask"][:, :1] = True
    with torch.no_grad():
        model.sync_in[2].w2.weight.normal_(std=0.03)
        out, _ = model(**data)
        audio, video = sync_deltas(model, data)
    assert out.shape == (2, 12, 64) and torch.isfinite(out).all()
    assert audio.shape == (2, 12, 32) and video.shape == (2, 6, 32)
    assert not torch.count_nonzero(audio[:, :2]) and not torch.count_nonzero(video[:, :1])
    print("[OK] same-seed common parameters are exact; distinct audio/video rates work")


def test_warmstart_and_padding():
    parent, model = make_pair()
    parent.eval(); model.eval()
    data = inputs()
    with torch.no_grad():
        expected, expected_ctc = parent(**{k: v for k, v in data.items() if not k.startswith("sync_")})
        actual, actual_ctc = model(**data)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
        for key in expected_ctc:
            torch.testing.assert_close(actual_ctc[key]["z_tilde"], expected_ctc[key]["z_tilde"], rtol=1e-5, atol=2e-6)
        model.sync_in[2].w2.weight.normal_(std=0.03)
        active, _ = model(**data)
        assert not torch.allclose(actual, active), "nonzero sync must affect flow"
        original = sync_deltas(model, data)
        inferred_mask = sync_deltas(model, {**data, "video_mask": None})
        for explicit, inferred in zip(original, inferred_mask):
            torch.testing.assert_close(explicit, inferred, rtol=0, atol=0)
        data["sync_feat"][1, 16:] = float("nan")
        padded = sync_deltas(model, data)
        for before, after in zip(original, padded):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            assert not torch.count_nonzero(after[:, :3]), "complementary prompt mask must remove sync"
            assert not torch.count_nonzero(after[1, 10:]), "padded destination frames must remain zero"
        poisoned, _ = model(**data)
        torch.testing.assert_close(poisoned, active, rtol=0, atol=0)
    print("[OK] zero-init warm start, nontrivial condition and poison-padding isolation")


def test_modulation_and_cfg():
    _, model = make_pair()
    model.eval()
    with torch.no_grad():
        model.sync_in[2].w2.weight.normal_(std=0.03)
    data, captures, handles = inputs(), {}, []
    for layer in range(4):
        def capture(_module, args, kwargs, layer=layer):
            captures[layer] = args[2 if layer == 0 else 1].detach().clone()
            if layer == 0:
                captures["video_t"] = kwargs["v_t"].detach().clone()
        handles.append(model.transformer_blocks[layer].register_forward_pre_hook(capture, with_kwargs=True))
    with torch.no_grad():
        model(**data)
        audio_sync, video_sync = sync_deltas(model, data)
        t = model.time_embed(data["time"])
        speaker = model.get_speaker_delta(data["speaker_embedding"], t)
        for layer in (0, 1):
            torch.testing.assert_close(captures[layer], t[:, None] + audio_sync)
        torch.testing.assert_close(captures["video_t"], t[:, None] + video_sync)
        for layer in (2, 3):
            torch.testing.assert_close(captures[layer], t[:, None] + audio_sync + speaker[:, None])
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        for flags in ({}, {"drop_text": True}, {"drop_video": True}):
            packed, _ = model(**data, cfg_infer=True, **flags)
            branches = [model(**data, **flags)[0]]
            if not flags:
                branches.append(model(**data, drop_video=True)[0])
            branches.append(model(**data, drop_audio_cond=True, drop_text=True, drop_video=True)[0])
            torch.testing.assert_close(packed, torch.cat(branches), rtol=1e-5, atol=2e-6)
        first, _ = model(**data, drop_video=True)
        second, _ = model(**{**data, "sync_feat": torch.randn_like(data["sync_feat"])}, drop_video=True)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    print("[OK] all-layer time modulation, speaker tail and B=2 packed/sequential CFG")


def test_cfm_backward_and_sampling():
    for checkpoint in (False, True):
        _, transformer = make_pair(checkpoint)
        model = CFM_VT(transformer=transformer, num_channels=64, audio_video_ratio=1, ctc_lambda=0.03,
                       audio_drop_prob=0.0, cond_drop_prob=0.0, text_drop_prob=0.0, video_drop_prob=0.0)
        data = inputs()
        kwargs = {"inp": data["x"], "text": data["text"], "video": data["video"],
                  "lens": torch.tensor([12, 10]), "text_lens": torch.tensor([4, 3]),
                  "video_lens": torch.tensor([12, 10]), "speaker_embedding": data["speaker_embedding"],
                  "sync_feat": data["sync_feat"], "sync_lens": data["sync_lens"]}
        with patch("aligndit.model.cfm_vt.random", return_value=0.5):
            loss, components, _, _ = model(**kwargs)
        assert torch.isfinite(loss) and "ctc_loss" in components
        loss.backward()
        gradient = transformer.sync_in[2].w2.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        with patch("aligndit.model.cfm_vt.random", return_value=0.5):
            model(**kwargs)[0].backward()
        for name, parameter in transformer.named_parameters():
            if name.startswith("sync_"):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
                assert torch.count_nonzero(parameter.grad), f"second update must reach {name}"
        for drop in ("cond_drop_prob", "video_drop_prob"):
            model.zero_grad(set_to_none=True)
            setattr(model, drop, 1.0)
            with patch("aligndit.model.cfm_vt.random", return_value=0.5):
                model(**kwargs)[0].backward()
            for name, parameter in transformer.named_parameters():
                if name.startswith("sync_"):
                    assert parameter.grad is not None and not torch.count_nonzero(parameter.grad), name
            setattr(model, drop, 0.0)
        sample_kwargs = {"cond": data["cond"][:, :3], "text": data["text"], "video": data["video"],
                         "speaker_embedding": data["speaker_embedding"], "sync_feat": data["sync_feat"],
                         "sync_lens": data["sync_lens"], "duration": torch.tensor([12, 10]),
                         "lens": torch.tensor([3, 3]), "steps": 1, "use_epss": False, "seed": 0}
        for guidance in (0.0, 1.0):
            out, _ = model.sample(**sample_kwargs, cfg_strength=guidance, cfg_strength_v=guidance)
            assert out.shape == (2, 12, 64) and torch.isfinite(out).all()
    print("[OK] CFM first/second gradients, checkpointing, joint visual dropout and ODE inference")


def test_required_features():
    _, model = make_pair()
    data = inputs()
    for bad in ({"sync_feat": None}, {"sync_lens": torch.tensor([24, 9])},
                {"sync_lens": torch.tensor([24, 32])}, {"sync_feat": torch.randn(2, 24, 12)}):
        try:
            model(**{**data, **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid feature contract accepted: {bad.keys()}")
    print("[OK] missing, invalid-width and invalid-length cache tensors fail clearly")


def main():
    torch.set_num_threads(1)
    test_valid_interpolation()
    test_same_seed_initialization_and_rates()
    test_warmstart_and_padding()
    test_modulation_and_cfg()
    test_cfm_backward_and_sampling()
    test_required_features()
    print("All Synchformer model contracts passed.")


if __name__ == "__main__":
    main()
