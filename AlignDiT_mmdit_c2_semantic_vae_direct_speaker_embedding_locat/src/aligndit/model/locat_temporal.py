"""LocAtViT-inspired, query-adaptive positive temporal attention enhancement.

This is a 1D cross-modal adaptation, not a copy of the paper's 2D module.
The Gaussian itself (not its logarithm) is added to attention logits. Widths
are standard deviations measured in seconds, and predictors are shared over
heads within one layer/direction. The stable constant initialization is our
pretrained-model adaptation choice, not LocAtViT's reported initialization.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LocAtTemporalBias(nn.Module):
    def __init__(
        self,
        dim_head: int,
        query_fps: float,
        key_fps: float,
        sigma_min_seconds: float = 0.025,
        sigma_max_seconds: float = 0.400,
        sigma_init_seconds: float = 0.100,
        alpha_init: float = 0.100,
        bias_mode: str = "gaussian",
    ):
        super().__init__()
        if type(dim_head) is not int or dim_head <= 0:
            raise ValueError("dim_head must be a positive integer")
        values = (query_fps, key_fps, sigma_min_seconds, sigma_max_seconds, sigma_init_seconds, alpha_init)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("LocAt temporal settings must be finite")
        if query_fps <= 0 or key_fps <= 0:
            raise ValueError("query_fps and key_fps must be positive")
        if not 0 < sigma_min_seconds < sigma_init_seconds < sigma_max_seconds:
            raise ValueError("expected 0 < sigma_min_seconds < sigma_init_seconds < sigma_max_seconds")
        if alpha_init <= 0:
            raise ValueError("alpha_init must be positive")
        if bias_mode not in {"gaussian", "uniform"}:
            raise ValueError("bias_mode must be 'gaussian' or 'uniform'")
        self.dim_head = dim_head
        self.query_fps = float(query_fps)
        self.key_fps = float(key_fps)
        self.sigma_min_seconds = float(sigma_min_seconds)
        self.sigma_max_seconds = float(sigma_max_seconds)
        self.sigma_init_seconds = float(sigma_init_seconds)
        self.alpha_init = float(alpha_init)
        self.bias_mode = bias_mode
        self.log_sigma = nn.Linear(dim_head, 1)
        self.log_alpha = nn.Linear(dim_head, 1)
        nn.init.zeros_(self.log_sigma.weight)
        nn.init.zeros_(self.log_alpha.weight)
        fraction = (sigma_init_seconds - sigma_min_seconds) / (sigma_max_seconds - sigma_min_seconds)
        nn.init.constant_(self.log_sigma.bias, math.log(fraction / (1.0 - fraction)))
        # Stable inverse of softplus, including large positive initial values.
        nn.init.constant_(self.log_alpha.bias, alpha_init + math.log(-math.expm1(-alpha_init)))
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    @staticmethod
    def _mask(mask, shape, device, name):
        if mask is None:
            return torch.ones(shape, dtype=torch.bool, device=device)
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must be bool")
        if tuple(mask.shape) != tuple(shape):
            raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(mask.shape)}")
        if mask.device != device:
            raise ValueError(f"{name} must be on {device}, got {mask.device}")
        return mask

    def _query_validity(self, batch, length, device, query_mask, enabled):
        valid = self._mask(query_mask, (batch, length), device, "query_mask")
        active = self._mask(enabled, (batch,), device, "enabled")
        return valid & active[:, None]

    def parameters_from_query(self, query, query_mask=None, enabled=None):
        """Predict sigma/alpha from normalized, PRE-RoPE Q; retain only scalars."""
        if query.ndim != 4 or query.shape[-1] != self.dim_head:
            raise ValueError(f"query must have shape [B,H,Q,{self.dim_head}]")
        valid = self._query_validity(query.shape[0], query.shape[2], query.device, query_mask, enabled)
        # Keep the bounded scale/positive amplitude and Gaussian arithmetic in
        # FP32 under mixed precision. Casts preserve gradients into Q/weights.
        with torch.autocast(device_type=query.device.type, enabled=False):
            q = query.float()
            sigma_logits = F.linear(q, self.log_sigma.weight.float(), self.log_sigma.bias.float())
            alpha_logits = F.linear(q, self.log_alpha.weight.float(), self.log_alpha.bias.float())
            sigma = self.sigma_min_seconds + (self.sigma_max_seconds - self.sigma_min_seconds) * sigma_logits.sigmoid()
            alpha = F.softplus(alpha_logits)
        with torch.no_grad():
            active = valid[:, None, :, None].expand_as(sigma)
            count = active.sum()

            def stats(value):
                value = value.detach()
                mean = value.masked_fill(~active, 0.0).sum() / count.clamp_min(1)
                minimum = value.masked_fill(~active, float("inf")).amin()
                maximum = value.masked_fill(~active, float("-inf")).amax()
                zero = value.new_zeros(())
                return mean, torch.where(count > 0, minimum, zero), torch.where(count > 0, maximum, zero)

            sigma_mean, sigma_min, sigma_max = stats(sigma)
            alpha_mean, alpha_min, alpha_max = stats(alpha)
            self.last_diagnostics = {
                "sigma_mean_seconds": sigma_mean,
                "sigma_min_seconds": sigma_min,
                "sigma_max_seconds": sigma_max,
                "alpha_mean": alpha_mean,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "valid_query_fraction": valid.float().mean(),
            }
        return sigma, alpha

    def bias_from_parameters(
        self, sigma, alpha, key_len, query_mask=None, key_mask=None, enabled=None, query_offset=0,
    ):
        if sigma.ndim != 4 or sigma.shape[-1] != 1 or sigma.shape != alpha.shape:
            raise ValueError("sigma and alpha must have matching shapes [B,H,Q,1]")
        if type(key_len) is not int or key_len <= 0:
            raise ValueError("key_len must be a positive integer")
        if type(query_offset) is not int or query_offset < 0:
            raise ValueError("query_offset must be a nonnegative integer")
        batch, _, query_len, _ = sigma.shape
        query_valid = self._query_validity(batch, query_len, sigma.device, query_mask, enabled)
        key_valid = self._mask(key_mask, (batch, key_len), sigma.device, "key_mask")
        with torch.autocast(device_type=sigma.device.type, enabled=False):
            if self.bias_mode == "uniform":
                # Keep a zero-valued sigma graph edge so this ablation does not
                # introduce extra unused parameters under DDP.
                bias = (alpha.float() + sigma.float() * 0.0).expand(-1, -1, -1, key_len)
            else:
                query_time = (torch.arange(query_len, device=sigma.device, dtype=torch.float32) + query_offset) / self.query_fps
                key_time = torch.arange(key_len, device=sigma.device, dtype=torch.float32) / self.key_fps
                delta = query_time[:, None] - key_time[None, :]
                bias = alpha.float() * torch.exp(-0.5 * (delta[None, None] / sigma.float()).square())
            valid = query_valid[:, None, :, None] & key_valid[:, None, None, :]
            return bias.masked_fill(~valid, 0.0)

    def forward(self, query, key_len, query_mask=None, key_mask=None, enabled=None, query_offset=0):
        sigma, alpha = self.parameters_from_query(query, query_mask=query_mask, enabled=enabled)
        return self.bias_from_parameters(
            sigma, alpha, key_len, query_mask=query_mask, key_mask=key_mask,
            enabled=enabled, query_offset=query_offset,
        ).to(query.dtype)
