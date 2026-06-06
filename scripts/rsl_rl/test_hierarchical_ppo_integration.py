"""End-to-end shape/logic test of the hierarchical PopArt PPO path (no IsaacSim).

Drives BonesCategoryRewardActorCritic + BonesCategoryRewardPPO through a full
rollout: act -> process_env_step -> compute_returns -> update, with fake obs
(TensorDict) and fake per-head rewards. Verifies shapes, that an optimizer step
runs, and that PopArt category stats get populated.

    conda activate env_isaaclab
    python scripts/rsl_rl/test_hierarchical_ppo_integration.py
"""

import os, sys
import torch
from tensordict import TensorDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "source", "whole_body_tracking"))
from whole_body_tracking.utils.hierarchical_popart import (
    BonesCategoryRewardActorCritic,
    BonesCategoryRewardPPO,
)

torch.manual_seed(0)
N, T = 64, 8
C, H = 3, 4
Dp, Dc, A = 12, 20, 5
device = "cpu"

obs_groups = {"policy": ["policy"], "critic": ["critic"]}
head_names = [f"head_{i}" for i in range(H)]
category_names = [f"cat_{i}" for i in range(C)]


def make_obs():
    return TensorDict(
        {
            "policy": torch.randn(N, Dp),
            "critic": torch.randn(N, Dc),
            "category": torch.randint(0, C, (N, 1)).float(),
        },
        batch_size=[N],
        device=device,
    )


obs = make_obs()
policy = BonesCategoryRewardActorCritic(
    obs, obs_groups, A,
    num_categories=C, num_heads=H,
    category_obs_group="category", category_names=category_names,
    actor_hidden_dims=[32, 32], critic_hidden_dims=[32, 32],
    popart_momentum=0.1,
)

# Forward checks.
v = policy.evaluate(obs)
assert v.shape == (N, H), f"evaluate shape {v.shape}"
v_un = policy.evaluate(obs, unnorm=True)
assert v_un.shape == (N, H), f"evaluate unnorm shape {v_un.shape}"
a = policy.act(obs)
assert a.shape == (N, A), f"act shape {a.shape}"
print(f"[forward] evaluate {tuple(v.shape)}, act {tuple(a.shape)} OK")

alg = BonesCategoryRewardPPO(
    policy,
    reward_weights=torch.ones(H),
    head_names=head_names,
    category_names=category_names,
    head_mode="per_term",
    per_head_rewards_key="per_head_rewards",
    category_obs_group="category",
    num_learning_epochs=2,
    num_mini_batches=4,
    learning_rate=1e-3,
    device=device,
    popart_actor_advantage_scaling="raw",
)
alg.init_storage("rl", N, T, obs, [A])

# Rollout.
for t in range(T):
    step_obs = make_obs()
    alg.act(step_obs)
    rewards = torch.randn(N)  # scalar env reward (unused by per-head path)
    dones = (torch.rand(N) < 0.1)
    extras = {
        "per_head_rewards": torch.randn(N, H) * torch.tensor([1.0, 50.0, 0.2, 10.0]),
        "time_outs": torch.zeros(N, dtype=torch.bool),
    }
    alg.process_env_step(step_obs, rewards, dones, extras)

assert alg.storage.step == T, f"storage filled {alg.storage.step}/{T}"

params_before = [p.detach().clone() for p in policy.parameters()]
alg.compute_returns(make_obs())

# Returns are normalized; values/returns buffers have shape [T,N,H].
assert alg.storage.returns.shape == (T, N, H), alg.storage.returns.shape
assert torch.isfinite(alg.storage.returns).all(), "non-finite returns"

losses = alg.update()
print(f"[update] losses = {losses}")
assert all(torch.isfinite(torch.tensor(v)) for v in losses.values()), "non-finite loss"

# An optimizer step actually changed parameters.
changed = any(not torch.equal(b, p) for b, p in zip(params_before, policy.parameters()))
assert changed, "optimizer step did not change parameters"

# PopArt category stats got populated for categories that appeared.
mean, std = policy.value_normalizer.mean_std()
assert mean.shape == (C, H) and std.shape == (C, H), (mean.shape, std.shape)
assert (std > 0).all(), "non-positive std"
print(f"[popart] mean[C,H]=\n{mean}\n std[C,H]=\n{std}")

# Diagnostics include per-(cat,head) keys.
diag_keys = [k for k in alg.last_diagnostics if k.startswith("popart/sigma/")]
print(f"[diagnostics] {len(diag_keys)} popart/sigma/* keys (expect {C*H})")
assert len(diag_keys) == C * H, diag_keys

print("\nHierarchical PopArt PPO integration test passed.")
