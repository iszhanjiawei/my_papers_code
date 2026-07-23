# Smoke test for the new DiT_VT_MMDiT backbone (dual-stream MM-DiT).
# Run: ~/ENTER/envs/aligndit/bin/python -u src/aligndit/script/misc/smoke_test_mmdit.py
# All tests run on CPU to avoid touching GPUs in use.

import copy
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../"))

from aligndit.model import CFM_VT, DiT_VT_MMDiT  # noqa: E402
from aligndit.model.backbone.dit_vt_mm import MMDiTBlock_VT  # noqa: E402
from aligndit.model.modules import DiTCrossBlock, MelSpec_tacotron  # noqa: E402


torch.manual_seed(0)

ARCH = dict(
    dim=768,
    depth=18,
    heads=12,
    ff_mult=2,
    text_dim=512,
    text_mask_padding=False,
    qk_norm="rms_norm",
    conv_layers=4,
    pe_attn_head=1,
    attn_mask_enabled=True,
    checkpoint_activations=False,
    use_conformer=True,
    layer_indices_ctc=[6, 12],
    n_mm_layers=12,
    audio_video_ratio=4,
    video_dim=1024,
    video_rope_scaled=True,
)
MEL = dict(
    target_sample_rate=16000,
    n_mel_channels=80,
    hop_length=160,
    win_length=640,
    n_fft=640,
    mel_spec_type="hifigan_16k",
)
VOCAB_SIZE = 100
CKPT_PATH = os.path.join(os.path.dirname(__file__), "../../../../ckpts/AlignDiT_pretrain_hifigan_16k_LibriSpeech_notext/model_500000.pt")


def build_model():
    transformer = DiT_VT_MMDiT(**ARCH, text_num_embeds=VOCAB_SIZE, mel_dim=MEL["n_mel_channels"])
    n_mm = sum(isinstance(b, MMDiTBlock_VT) for b in transformer.transformer_blocks)
    n_single = sum(
        isinstance(b, DiTCrossBlock) and not isinstance(b, MMDiTBlock_VT) for b in transformer.transformer_blocks
    )
    assert n_mm == ARCH["n_mm_layers"] and n_mm + n_single == ARCH["depth"], (n_mm, n_single)
    print(f"[OK] model built: {n_mm} MM blocks + {n_single} single-stream blocks")
    n_params = sum(p.numel() for p in transformer.parameters())
    print(f"     transformer params: {n_params / 1e6:.1f}M")

    mel_kwargs = {k: v for k, v in MEL.items() if k != "mel_spec_type"}
    model = CFM_VT(
        transformer=transformer,
        mel_spec_module=MelSpec_tacotron(**MEL),
        mel_spec_kwargs=mel_kwargs,
        vocab_char_map=None,
        ctc_lambda=0.1,
    )
    return model


def _assert_nonzero_finite_grad(name, grad):
    assert grad is not None, f"missing gradient for {name}"
    assert torch.isfinite(grad).all(), f"non-finite gradient for {name}"
    assert grad.abs().sum().item() > 0, f"zero gradient for {name}"


def test_mm_gated_branches_wake_up(model):
    """A zero gate must open first, then allow gradients into its attention branch."""
    block = copy.deepcopy(model.transformer.transformer_blocks[0]).eval()
    dim = ARCH["dim"]
    text_dim = ARCH["text_dim"]

    assert torch.count_nonzero(block.cross_attn_ada.weight) == 0
    assert torch.count_nonzero(block.v_attn_norm.linear.weight) == 0
    assert torch.count_nonzero(block.cross_attn.out_proj.weight) > 0
    assert torch.count_nonzero(block.v_attn.to_out[0].weight) > 0

    batch, audio_len, video_len, text_len = 2, 24, 6, 10
    x = torch.randn(batch, audio_len, dim)
    v = torch.randn(batch, video_len, dim)
    t = torch.randn(batch, dim)
    text = torch.randn(batch, text_len, text_dim)
    audio_mask = torch.ones(batch, audio_len, dtype=torch.bool)
    video_mask = torch.ones(batch, video_len, dtype=torch.bool)
    text_mask = torch.ones(batch, text_len, dtype=torch.bool)
    generation_mask = torch.ones(batch, audio_len, dtype=torch.bool)
    x_probe = torch.randn_like(x)
    v_probe = torch.randn_like(v)

    def branch_loss():
        x_out, v_out = torch.utils.checkpoint.checkpoint(
            model.transformer.ckpt_wrapper(block),
            x,
            v,
            t,
            audio_mask,
            video_mask,
            None,
            None,
            text,
            text_mask,
            generation_mask,
            use_reentrant=False,
        )
        return (x_out * x_probe).mean() + (v_out * v_probe).mean()

    optimizer = torch.optim.SGD(block.parameters(), lr=0.1)

    # Update 1: normally initialized branch outputs provide gradients to the zero gates.
    optimizer.zero_grad(set_to_none=True)
    branch_loss().backward()
    _assert_nonzero_finite_grad(
        "MM text cross-attention gate",
        block.cross_attn_ada.weight.grad[2 * dim : 3 * dim],
    )
    _assert_nonzero_finite_grad(
        "MM video-attention gate",
        block.v_attn_norm.linear.weight.grad[2 * dim : 3 * dim],
    )
    optimizer.step()

    # Update 2: opened gates propagate gradients into the complete attention branches.
    optimizer.zero_grad(set_to_none=True)
    branch_loss().backward()
    _assert_nonzero_finite_grad("MM text cross-attention output", block.cross_attn.out_proj.weight.grad)
    _assert_nonzero_finite_grad("MM text cross-attention query", block.cross_attn.q_proj_weight.grad)
    _assert_nonzero_finite_grad("MM video-attention output", block.v_attn.to_out[0].weight.grad)
    _assert_nonzero_finite_grad("MM video-attention query", block.v_attn.to_q.weight.grad)
    print("[OK] gated MM text/video attention branches wake up and receive non-zero gradients")


def test_train_forward_backward(model):
    model.train()
    b, n, nv, nt = 2, 200, 50, 20
    mel = torch.randn(b, n, MEL["n_mel_channels"])
    video = torch.randn(b, nv, ARCH["video_dim"])
    text = torch.randint(1, VOCAB_SIZE, (b, nt))
    lens = torch.tensor([n, n - 40])
    text_lens = torch.tensor([nt, nt - 5])
    video_lens = torch.tensor([nv, (n - 40) // 4])

    captured_generation_masks = []

    def capture_generation_mask(_module, _args, kwargs):
        captured_generation_masks.append(kwargs["generation_mask"].detach().clone())

    hook = model.transformer.transformer_blocks[0].register_forward_pre_hook(
        capture_generation_mask, with_kwargs=True
    )
    try:
        loss, component_losses, cond, pred = model(
            mel, text, video, lens=lens, text_lens=text_lens, video_lens=video_lens
        )
    finally:
        hook.remove()

    expected_generation_mask = torch.all(cond == 0, dim=-1)
    assert len(captured_generation_masks) == 1
    assert torch.equal(captured_generation_masks[0], expected_generation_mask)
    assert torch.all(captured_generation_masks[0].sum(dim=-1) > 0)
    assert torch.isfinite(loss), f"loss is not finite: {loss}"
    assert pred.shape == mel.shape, (pred.shape, mel.shape)
    loss.backward()

    # gradients must reach pretrained-path (audio stream) and new video-stream params
    tf = model.transformer
    checks = {
        "audio attn (pretrained path)": tf.transformer_blocks[0].attn.to_q.weight,
        "video stream attn": tf.transformer_blocks[0].v_attn.to_q.weight,
        "video stream adaLN": tf.transformer_blocks[0].v_attn_norm.linear.weight,
        "text cross attn": tf.transformer_blocks[0].cross_attn.q_proj_weight,
        "single-stream block attn": tf.transformer_blocks[-1].attn.to_q.weight,
        "video input proj": tf.video_embed.proj.weight,
        "audio input proj": tf.input_embed.proj.weight,
        "ctc projector": tf.projectors_ctc[0].model[0].weight,
    }
    for name, p in checks.items():
        assert p.grad is not None and torch.isfinite(p.grad).all(), f"no/invalid grad for {name}"
    print(f"[OK] train forward/backward: loss={loss.item():.4f}, components={component_losses}")
    print("[OK] training rand_span_mask reaches MM-DiT as the explicit generation_mask")
    model.zero_grad(set_to_none=True)


def test_modality_drop(model):
    model.train()
    b, n, nv, nt = 2, 120, 30, 12
    mel = torch.randn(b, n, MEL["n_mel_channels"])
    video = torch.randn(b, nv, ARCH["video_dim"])
    text = torch.randint(1, VOCAB_SIZE, (b, nt))

    # exercise drop branches directly on the backbone
    time = torch.rand(b)
    x = torch.randn(b, n, MEL["n_mel_channels"])
    cond = torch.randn(b, n, MEL["n_mel_channels"])
    mask = torch.ones(b, n, dtype=torch.bool)
    text_mask = torch.ones(b, nt, dtype=torch.bool)
    video_mask = torch.ones(b, nv, dtype=torch.bool)
    complementary_mask = torch.ones(b, nv, dtype=torch.bool)
    generation_mask = torch.ones(b, n, dtype=torch.bool)
    for drop_text, drop_video, drop_audio_cond in [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ]:
        pred, inter = model.transformer(
            x=x,
            cond=cond,
            text=text,
            video=video,
            time=time,
            mask=mask,
            text_mask=text_mask,
            video_mask=video_mask,
            complementary_mask=complementary_mask,
            generation_mask=generation_mask,
            drop_audio_cond=drop_audio_cond,
            drop_text=drop_text,
            drop_video=drop_video,
        )
        assert pred.shape == (b, n, MEL["n_mel_channels"])
        assert torch.isfinite(pred).all()
        assert set(inter.keys()) == set(ARCH["layer_indices_ctc"])
    print("[OK] modality drop branches (cond / drop_text / drop_video / uncond)")

    _ = mel  # unused


def test_sample(model):
    model.eval()
    n_prompt, dur, nv = 100, 200, 50
    cond = torch.randn(1, n_prompt, MEL["n_mel_channels"])
    cond[:, :8] = 0  # real prompt silence must remain outside the generated region
    video = torch.randn(1, nv, ARCH["video_dim"])
    text = torch.randint(1, VOCAB_SIZE, (1, 24))
    captured_generation_masks = []

    def capture_generation_mask(_module, _args, kwargs):
        captured_generation_masks.append(kwargs["generation_mask"].detach().clone())

    hook = model.transformer.transformer_blocks[0].register_forward_pre_hook(
        capture_generation_mask, with_kwargs=True
    )
    try:
        with torch.no_grad():
            out, trajectory = model.sample(
                cond=cond,
                text=text,
                duration=dur,
                video=video,
                steps=2,
                cfg_strength=2.0,
                cfg_strength_v=2.0,
                use_epss=False,
            )
    finally:
        hook.remove()

    expected_generation_mask = torch.zeros(dur, dtype=torch.bool)
    expected_generation_mask[n_prompt:] = True
    assert len(captured_generation_masks) > 0
    for generation_mask in captured_generation_masks:
        assert generation_mask.shape == (3, dur)
        assert torch.equal(generation_mask, expected_generation_mask.expand_as(generation_mask))
    assert out.shape[0] == 1 and out.shape[2] == MEL["n_mel_channels"], out.shape
    assert torch.isfinite(out).all()
    print(f"[OK] sample (multimodal CFG, 3-branch): out shape={tuple(out.shape)}")

    # TTS / VTS single-modality guidance paths
    for ignore in ["text", "video"]:
        model.transformer.clear_cache()
        with torch.no_grad():
            out, _ = model.sample(
                cond=cond,
                text=text,
                duration=dur,
                video=video,
                steps=2,
                cfg_strength=2.0,
                cfg_strength_v=2.0,
                use_epss=False,
                ignore_modality=ignore,
            )
        assert torch.isfinite(out).all()
    print("[OK] sample with ignore_modality=text/video (2-branch CFG)")


def test_pretrained_ckpt_compat(model):
    if not os.path.exists(CKPT_PATH):
        print(f"[SKIP] pretrained ckpt not found: {CKPT_PATH}")
        return
    checkpoint = torch.load(CKPT_PATH, weights_only=True, map_location="cpu")
    ckpt_sd = {
        k.replace("ema_model.", ""): v
        for k, v in checkpoint["ema_model_state_dict"].items()
        if k not in ["initted", "update", "step"]
    }
    model_sd = model.state_dict()

    matched, mismatched, extra = [], [], []
    for k, v in ckpt_sd.items():
        if k not in model_sd:
            extra.append(k)
        elif v.shape != model_sd[k].shape:
            mismatched.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
        else:
            matched.append(k)
    missing = [k for k in model_sd if k not in ckpt_sd]

    print(f"[INFO] ckpt keys matched: {len(matched)}, shape-mismatch: {len(mismatched)}, "
          f"ckpt-only (ignored): {len(extra)}, model-only (new params): {len(missing)}")

    assert len(mismatched) == 0, f"shape mismatches: {mismatched[:5]}"

    # every audio-path parameter of the pretrained DiT must be matched
    must_match_prefixes = [
        "transformer.time_embed.",
        "transformer.input_embed.proj.",
        "transformer.input_embed.conv_pos_embed.",
        "transformer.norm_out.",
        "transformer.proj_out.",
    ]
    for i in range(ARCH["depth"]):
        must_match_prefixes += [
            f"transformer.transformer_blocks.{i}.attn_norm.",
            f"transformer.transformer_blocks.{i}.attn.",
            f"transformer.transformer_blocks.{i}.ff.",
        ]
    matched_set = set(matched)
    for prefix in must_match_prefixes:
        ckpt_keys = [k for k in ckpt_sd if k.startswith(prefix)]
        assert len(ckpt_keys) > 0, f"no ckpt keys under {prefix}"
        for k in ckpt_keys:
            assert k in matched_set, f"pretrained key not loadable: {k}"

    # sanity: new params are exactly the video stream / text cross-attn / video embed / ctc projectors
    allowed_new_prefixes = ("transformer.transformer_blocks.", "transformer.video_embed.", "transformer.projectors_ctc.", "transformer.text_embed.")
    for k in missing:
        assert k.startswith(allowed_new_prefixes), f"unexpected new param: {k}"
        if k.startswith("transformer.transformer_blocks."):
            assert (".v_attn" in k or ".v_ff" in k or ".cross_attn" in k), f"unexpected new block param: {k}"

    # simulate the _safe_merge load and verify it works end to end
    merged = {k: (ckpt_sd[k] if k in matched_set else model_sd[k]) for k in model_sd}
    model.load_state_dict(merged)
    print("[OK] pretrained checkpoint merge-load (audio path fully covered, new params kept)")


def main():
    torch.set_num_threads(8)
    model = build_model()
    test_mm_gated_branches_wake_up(model)
    test_train_forward_backward(model)
    test_modality_drop(model)
    test_pretrained_ckpt_compat(model)
    test_sample(model)
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
