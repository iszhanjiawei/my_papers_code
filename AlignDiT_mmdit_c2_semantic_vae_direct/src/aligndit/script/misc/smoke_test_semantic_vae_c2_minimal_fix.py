"""CPU smoke tests for the Semantic-VAE C2 minimal numerical repair."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from aligndit.model.backbone.dit_vt_mm import DiT_VT_MMDiT
from aligndit.model.trainer_semantic_vae_minimal_fix import scale_safe_global_grad_norm


PROJECT_ROOT = Path(__file__).resolve().parents[4]


def test_parameter_free_text_context_normalization() -> None:
    module = object.__new__(DiT_VT_MMDiT)
    nn.Module.__init__(module)
    module.last_text_context_raw_rms = None
    module.last_text_context_post_rms = None

    generator = torch.Generator().manual_seed(666)
    text = torch.randn(2, 5, 8, generator=generator) * 1000
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]], dtype=torch.bool)
    state_before = tuple(module.state_dict())
    normalized, normalized_mask = module._stabilize_text_context(text, mask)

    valid = normalized[normalized_mask]
    torch.testing.assert_close(valid.mean(dim=-1), torch.zeros(5), atol=2e-5, rtol=0)
    torch.testing.assert_close(valid.square().mean(dim=-1), torch.ones(5), atol=2e-5, rtol=0)
    assert torch.count_nonzero(normalized[~normalized_mask]).item() == 0
    assert module.last_text_context_raw_rms.item() > 100
    assert 0.999 < module.last_text_context_post_rms.item() < 1.001
    assert tuple(module.state_dict()) == state_before


def test_scale_safe_gradient_norm() -> None:
    first = nn.Parameter(torch.zeros(3))
    second = nn.Parameter(torch.zeros(4))
    first.grad = torch.tensor([3.0, 4.0, 0.0])
    second.grad = torch.tensor([0.0, 0.0, 0.0, 12.0])
    assert scale_safe_global_grad_norm((first, second)) == 13.0

    first.grad.fill_(1e20)
    second.grad.fill_(1e20)
    large_norm = scale_safe_global_grad_norm((first, second))
    assert math.isfinite(large_norm) and large_norm > 1e20

    first.grad[0] = float("inf")
    assert math.isnan(scale_safe_global_grad_norm((first, second)))


def test_minimal_config_diff_is_explicit() -> None:
    from omegaconf import OmegaConf

    config_dir = PROJECT_ROOT / "src/aligndit/config"
    direct = OmegaConf.load(config_dir / "finetune_celebvdub_mm_c2_semantic_vae_direct.yaml")
    repaired = OmegaConf.load(config_dir / "finetune_celebvdub_mm_c2_semantic_vae_minimal_fix.yaml")
    allowed_changes = {
        "seed",
        "optim.max_updates",
        "optim.learning_rate",
        "monitoring.global_grad_norm_min_threshold",
        "monitoring.global_grad_norm_abort_threshold",
        "monitoring.post_text_rms_min",
        "monitoring.post_text_rms_max",
        "model.name",
        "model.arch.normalize_text_context",
        "ckpts.save_dir",
    }

    def flatten(value, prefix=""):
        if not OmegaConf.is_config(value):
            return {prefix: value}
        container: Any = OmegaConf.to_container(value, resolve=False)
        if isinstance(container, list):
            return {prefix: container}
        flattened = {}
        for key, child in container.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(child, dict):
                flattened.update(flatten(OmegaConf.create(child), path))
            else:
                flattened[path] = child
        return flattened

    direct_flat = flatten(direct)
    repaired_flat = flatten(repaired)
    changed = {
        key
        for key in set(direct_flat) | set(repaired_flat)
        if direct_flat.get(key, object()) != repaired_flat.get(key, object())
    }
    assert changed == allowed_changes, f"unexpected Direct-to-minimal config differences: {sorted(changed)}"


def main() -> None:
    test_parameter_free_text_context_normalization()
    test_scale_safe_gradient_norm()
    test_minimal_config_diff_is_explicit()
    print("Semantic-VAE C2 minimal-fix CPU smoke tests passed")


if __name__ == "__main__":
    main()
