"""ActorCriticPopArt — multi-head critic with DiagonalPopArt value normalization.

Subclasses RSL-RL's `ActorCritic`. Replaces the critic's final
`Linear(_, 1)` with `Linear(_, K)` and wraps it in a `DiagonalPopArt` so
each category gets its own running `(μ_k, σ_k)`.

Two forward paths on the critic:

- `evaluate(obs)` returns scalar `[N, 1]` in **unnormalized** space —
  forward → unnorm via PopArt → gather active column using
  `obs["category"]`. This is what RSL-RL's rollout (`act`),
  `compute_returns` (`last_values`), and `RolloutStorage.values` consume,
  so the rest of the algorithm sees scalar values exactly as in vanilla
  PPO.

- `evaluate_columns(obs)` returns the raw `[N, K]` **normalized** critic
  output. Used inside `PPOPopArt.update` to compute the masked critic loss
  in normalized space against `(returns − μ_k)/σ_k`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.modules import ActorCritic

from whole_body_tracking.utils.popart import DiagonalPopArt


class ActorCriticPopArt(ActorCritic):
    def __init__(
        self,
        obs,
        obs_groups,
        num_actions: int,
        *,
        num_categories: int,
        popart_momentum: float = 0.1,
        category_obs_group: str = "category",
        category_names: list[str] | None = None,
        **kwargs,
    ):
        # Build the standard ActorCritic with a *single*-output critic so all
        # parent init logic (MLP construction, normalizers, distribution) is
        # untouched. Then swap in a K-output last layer and wrap with PopArt.
        super().__init__(obs, obs_groups, num_actions, **kwargs)

        if num_categories < 2:
            raise ValueError(f"num_categories must be >= 2, got {num_categories}")
        if category_obs_group not in obs:
            raise ValueError(
                f"category_obs_group {category_obs_group!r} not present in obs. "
                f"Available groups: {list(obs.keys())}"
            )

        self.num_categories = num_categories
        self.category_obs_group = category_obs_group
        self.category_names = (
            list(category_names) if category_names is not None
            else [f"cat_{i}" for i in range(num_categories)]
        )
        if len(self.category_names) < num_categories:
            # Pad with cat_<i> for any missing names so indexing is always safe.
            self.category_names += [
                f"cat_{i}" for i in range(len(self.category_names), num_categories)
            ]

        # The parent built `self.critic = MLP(_, 1, hidden, activation)`.
        # MLP is an nn.Sequential, last module is Linear(in, 1).
        old_last: nn.Linear = self.critic[-1]
        if not isinstance(old_last, nn.Linear) or old_last.out_features != 1:
            raise RuntimeError(
                "Expected critic's last layer to be Linear(_, 1) — found "
                f"{type(old_last).__name__} with out_features="
                f"{getattr(old_last, 'out_features', '?')}"
            )
        new_last = nn.Linear(old_last.in_features, num_categories, bias=True)
        nn.init.zeros_(new_last.bias)
        nn.init.orthogonal_(new_last.weight, gain=1.0)
        self.critic[-1] = new_last

        # PopArt holds references to the new layer's parameters and applies
        # the ART correction in-place during `update_masked`.
        self.value_normalizer = DiagonalPopArt(
            num_categories,
            new_last.weight,
            new_last.bias,
            momentum=popart_momentum,
        )
        # Make the K-head visible in the startup log. RSL-RL's parent
        # ActorCritic.__init__ prints the critic before our override fires,
        # so without this line the log misleadingly shows Linear(_, 1).
        print(
            f"[ActorCriticPopArt] Critic last layer replaced: "
            f"Linear({old_last.in_features}, {num_categories})  "
            f"PopArt momentum={popart_momentum}, "
            f"category_obs_group={category_obs_group!r}"
        )

    # ── Forward paths ────────────────────────────────────────────────────

    def _critic_forward(self, obs) -> torch.Tensor:
        """Raw normalized K-vector critic output [N, K]."""
        critic_in = self.get_critic_obs(obs)
        critic_in = self.critic_obs_normalizer(critic_in)
        return self.critic(critic_in)

    def _gather_category(self, obs) -> torch.Tensor:
        """Per-env category index, [N], long."""
        cat = obs[self.category_obs_group]
        if cat.dim() == 2:
            cat = cat.squeeze(-1)
        return cat.long()

    def evaluate(self, obs, **kwargs) -> torch.Tensor:
        """Scalar [N, 1] in unnormalized space, gathered for the active
        category. This is the method RSL-RL's rollout / GAE call."""
        v_K_norm = self._critic_forward(obs)                    # [N, K]
        v_K = self.value_normalizer(v_K_norm, unnorm=True)      # [N, K]
        cat = self._gather_category(obs)                        # [N]
        v = v_K.gather(-1, cat.unsqueeze(-1))                   # [N, 1]
        return v

    def evaluate_columns(self, obs, **kwargs) -> torch.Tensor:
        """Raw normalized K-vector critic output. Used by `PPOPopArt.update`."""
        return self._critic_forward(obs)
