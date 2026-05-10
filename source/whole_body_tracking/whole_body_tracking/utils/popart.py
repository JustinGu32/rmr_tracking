"""DiagonalPopArt — per-channel adaptive value normalization.

The `DiagonalPopArt` class is copied verbatim from
xupei0610/CompositeMotion's `models.py:42–83`. The class wraps the final
linear layer of a critic; calling `update(returns)` adjusts running
mean/variance estimates per channel and applies the ART correction to
`weight`/`bias` so that the *unnormalized* critic predictions are preserved
across the stats shift.

Two PopArt-specific extensions are added on top of the verbatim copy:

- `mean_std(active_only=False)` — return debiased `(mean, std)` per channel,
  used to normalize per-sample returns inside the PPO update.
- `update_masked(returns_flat, cat_flat, K)` — Option A from
  `popart_plan.md` §5.6. For each category index `k`, only the rows where
  `cat_flat == k` contribute to column `k`'s stats. ART correction fires
  per-column. Categories absent from this rollout are left untouched.

The verbatim `update(x)` is preserved for any future single-channel use.
"""

from __future__ import annotations

import torch


class DiagonalPopArt(torch.nn.Module):
    """Per-channel adaptive value rescaling. Owns references to the wrapped
    layer's `weight` and `bias` so the ART correction can be applied
    in-place."""

    def __init__(self, dim: int, weight: torch.Tensor, bias: torch.Tensor, momentum: float = 0.1):
        super().__init__()
        self.epsilon = 1e-5

        self.momentum = momentum
        self.register_buffer("m", torch.zeros((dim,), dtype=torch.float64))
        self.register_buffer("v", torch.full((dim,), self.epsilon, dtype=torch.float64))
        self.register_buffer("debias", torch.zeros(1, dtype=torch.float64))

        # NOTE: weight / bias are references to the wrapped Linear's parameters.
        # Stored as plain attributes (not Parameters) so PyTorch doesn't think
        # this module owns them. Their updates happen in-place inside `update`.
        self.weight = weight
        self.bias = bias

    def forward(self, x, unnorm: bool = False):
        debias = self.debias.clip(min=self.epsilon)
        mean = self.m / debias
        var = (self.v - self.m.square()).div_(debias)
        if unnorm:
            std = torch.sqrt(var)
            return (mean + std * x).to(x.dtype)
        x = ((x - mean) * torch.rsqrt(var)).to(x.dtype)
        return x

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        """Update stats from a flat batch `[N, dim]`. All rows contribute to
        all columns — fine when every channel sees every row (composite-reward
        setup). Use `update_masked` for the per-category multi-task setup."""
        x = x.view(-1, x.size(-1))
        running_m = torch.mean(x, dim=0)
        running_v = torch.mean(x.square(), dim=0)
        new_m = self.m.mul(1 - self.momentum).add_(running_m, alpha=self.momentum)
        new_v = self.v.mul(1 - self.momentum).add_(running_v, alpha=self.momentum)
        std = (self.v - self.m.square()).sqrt_()
        new_std_inv = (new_v - new_m.square()).rsqrt_()

        scale = std.mul_(new_std_inv)
        shift = (self.m - new_m).mul_(new_std_inv)

        self.bias.data.mul_(scale).add_(shift)
        self.weight.data.mul_(scale.unsqueeze_(-1))

        self.debias.data.mul_(1 - self.momentum).add_(1.0 * self.momentum)
        self.m.data.copy_(new_m)
        self.v.data.copy_(new_v)

    @torch.no_grad()
    def update_masked(self, returns_flat: torch.Tensor, cat_flat: torch.Tensor, K: int):
        """Per-category masked stats update + ART correction.

        Args:
            returns_flat: scalar returns, shape `[N]`.
            cat_flat:     long category indices, shape `[N]`, values in `[0, K)`.
            K:            number of categories (must equal `self.m.numel()`).

        For each category `k`, compute `running_m_k = mean(returns[mask_k])`
        and `running_v_k = mean(returns[mask_k]**2)`, then EMA-update column
        `k`'s `(m, v)` and apply per-column ART correction to the wrapped
        layer's row `k` of `weight` and column `k` of `bias`. Categories with
        no rows in this rollout are left untouched (no EMA, no ART).
        """
        assert K == self.m.numel(), f"K={K} mismatches stats dim={self.m.numel()}"
        device = self.m.device

        running_m = torch.zeros(K, dtype=self.m.dtype, device=device)
        running_v = torch.zeros(K, dtype=self.v.dtype, device=device)
        has_data = torch.zeros(K, dtype=torch.bool, device=device)

        # Cast to float64 to match buffer dtype for stat math.
        r64 = returns_flat.to(self.m.dtype)
        for k in range(K):
            mask = cat_flat == k
            n_k = int(mask.sum().item())
            if n_k > 0:
                rk = r64[mask]
                running_m[k] = rk.mean()
                running_v[k] = rk.square().mean()
                has_data[k] = True

        # EMA — only update columns that saw data this rollout.
        new_m = self.m.clone()
        new_v = self.v.clone()
        new_m[has_data] = self.m[has_data].mul(1 - self.momentum).add_(running_m[has_data], alpha=self.momentum)
        new_v[has_data] = self.v[has_data].mul(1 - self.momentum).add_(running_v[has_data], alpha=self.momentum)

        # ART correction. Compute scale / shift for every column, then mask
        # inactive columns to identity (scale=1, shift=0) so they're untouched.
        std = (self.v - self.m.square()).clamp_min_(self.epsilon).sqrt_()
        new_std_inv = (new_v - new_m.square()).clamp_min_(self.epsilon).rsqrt_()
        scale = std * new_std_inv  # [K]
        shift = (self.m - new_m) * new_std_inv  # [K]
        scale = torch.where(has_data, scale, torch.ones_like(scale))
        shift = torch.where(has_data, shift, torch.zeros_like(shift))

        # Apply per-column to the wrapped layer.
        self.bias.data.mul_(scale.to(self.bias.dtype)).add_(shift.to(self.bias.dtype))
        self.weight.data.mul_(scale.unsqueeze(-1).to(self.weight.dtype))

        # Tick global debias if at least one category contributed.
        if bool(has_data.any().item()):
            self.debias.data.mul_(1 - self.momentum).add_(1.0 * self.momentum)
        self.m.data.copy_(new_m)
        self.v.data.copy_(new_v)

    @torch.no_grad()
    def mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Debiased per-channel `(mean, std)`, both shape `[K]`, float32."""
        debias = self.debias.clip(min=self.epsilon)
        mean = (self.m / debias).to(torch.float32)
        var = ((self.v - self.m.square()) / debias).clamp_min(self.epsilon).to(torch.float32)
        std = var.sqrt()
        return mean, std
