"""Standalone numerical tests for DiagonalCategoryRewardPopArt.

Run (no IsaacSim needed):
    conda activate env_isaaclab
    python scripts/rsl_rl/test_hierarchical_popart.py

Verifies:
  1. ART correction preserves the *unnormalized* critic prediction per (cat,head)
     across a stats update (the core PopArt invariant).
  2. update_masked only touches categories that received >= min_samples samples.
  3. forward(normalize) and forward(unnormalize) are inverses given current stats.
  4. Per-category stats actually converge toward the per-category return means.
"""

import sys, os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "whole_body_tracking"))
from whole_body_tracking.utils.hierarchical_popart import DiagonalCategoryRewardPopArt


def make_normalizer(C, H, in_dim, momentum=0.1):
    head = torch.nn.Linear(in_dim, C * H)
    torch.nn.init.uniform_(head.weight, -0.1, 0.1)
    torch.nn.init.uniform_(head.bias, -0.1, 0.1)
    norm = DiagonalCategoryRewardPopArt(C, H, head.weight, head.bias, momentum=momentum, min_samples=2)
    return head, norm


def unnorm_pred(head, norm, feat, cat):
    """Unnormalized critic prediction for active category, [N,H]."""
    all_norm = head(feat).view(-1, norm.C, norm.H)
    active = all_norm[torch.arange(feat.shape[0]), cat]
    return norm(active, cat, unnorm=True)


def test_art_preserves_output():
    torch.manual_seed(0)
    C, H, in_dim, N = 4, 3, 8, 4096
    head, norm = make_normalizer(C, H, in_dim)
    feat = torch.randn(N, in_dim)
    cat = torch.randint(0, C, (N,))

    # Warm up stats so they are non-trivial.
    for _ in range(5):
        returns = torch.randn(N, H) * torch.tensor([2.0, 50.0, 0.3]) + torch.tensor([1.0, -20.0, 0.0])
        norm.update_masked(returns, cat)

    before = unnorm_pred(head, norm, feat, cat).clone()
    # Another big stats shift.
    returns = torch.randn(N, H) * torch.tensor([5.0, 80.0, 1.0]) + torch.tensor([3.0, 40.0, -2.0])
    norm.update_masked(returns, cat)
    after = unnorm_pred(head, norm, feat, cat)

    max_abs = (after - before).abs().max().item()
    rel = max_abs / before.abs().mean().clamp_min(1e-6).item()
    print(f"[art_preserves_output] max|Δ unnorm pred| = {max_abs:.3e}  (rel {rel:.3e})")
    assert max_abs < 1e-3, f"ART correction did not preserve output: {max_abs}"


def test_masked_update_isolation():
    torch.manual_seed(1)
    C, H, in_dim, N = 5, 2, 4, 2000
    head, norm = make_normalizer(C, H, in_dim)
    # Only categories 1 and 3 receive data.
    cat = torch.where(torch.rand(N) < 0.5, torch.full((N,), 1), torch.full((N,), 3))
    returns = torch.randn(N, H) * 10 + 5
    mean_before = norm.mean.clone()
    debias_before = norm.debias.clone()
    norm.update_masked(returns, cat)
    touched = (norm.mean != mean_before).any(dim=1)
    debias_touched = norm.debias != debias_before
    print(f"[masked_update_isolation] categories touched (mean): {touched.tolist()}")
    print(f"[masked_update_isolation] categories touched (debias): {debias_touched.tolist()}")
    expected = torch.zeros(C, dtype=torch.bool); expected[[1, 3]] = True
    assert torch.equal(touched, expected), "Wrong categories updated (mean)"
    assert torch.equal(debias_touched, expected), "Wrong categories updated (debias)"


def test_forward_inverse():
    torch.manual_seed(2)
    C, H, in_dim, N = 3, 4, 4, 4096
    head, norm = make_normalizer(C, H, in_dim)
    cat = torch.randint(0, C, (N,))
    for _ in range(10):
        norm.update_masked(torch.randn(N, H) * 7 + 2, cat)
    v = torch.randn(N, H)
    roundtrip = norm(norm(v, cat, unnorm=False), cat, unnorm=True)
    err = (roundtrip - v).abs().max().item()
    print(f"[forward_inverse] max|Δ| = {err:.3e}")
    assert err < 1e-4, f"normalize/unnormalize not inverse: {err}"


def test_stats_converge():
    torch.manual_seed(3)
    C, H, in_dim, N = 3, 2, 4, 8192
    head, norm = make_normalizer(C, H, in_dim, momentum=0.05)
    cat = torch.randint(0, C, (N,))
    true_mean = torch.tensor([[1.0, 100.0], [-5.0, 0.5], [10.0, -50.0]])
    true_std = torch.tensor([[2.0, 20.0], [1.0, 0.2], [5.0, 8.0]])
    for _ in range(400):
        returns = torch.randn(N, H) * true_std[cat] + true_mean[cat]
        norm.update_masked(returns, cat)
    mean, std = norm.mean_std()
    mean_err = (mean - true_mean).abs().max().item()
    std_err = (std - true_std).abs().max().item()
    print(f"[stats_converge] max|Δmean| = {mean_err:.3f}  max|Δstd| = {std_err:.3f}")
    print(f"  recovered mean=\n{mean}\n  recovered std=\n{std}")
    assert mean_err < 1.0, f"means did not converge: {mean_err}"
    assert std_err < 2.0, f"stds did not converge: {std_err}"


if __name__ == "__main__":
    test_art_preserves_output()
    test_masked_update_isolation()
    test_forward_inverse()
    test_stats_converge()
    print("\nAll hierarchical PopArt numerical tests passed.")
