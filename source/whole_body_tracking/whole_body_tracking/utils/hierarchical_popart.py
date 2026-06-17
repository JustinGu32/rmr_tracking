"""Hierarchical (motion-category × reward-head) PopArt for the BONES generalist.

This combines the two existing PopArt variants:

- **Category-conditioned PopArt** (justingu's ``popart.py`` / ``ppo_popart.py``):
  separate running ``(mu, sigma)`` per *motion category* (walk / static / jump …),
  selected per-env from ``obs["category"]``.
- **Reward-head / micro PopArt** (this repo's ``bones_popart.py``): separate
  ``(mu, sigma)`` per *reward head/group* (upper_body / lower_body / global /
  regularization …), with per-head GAE and grouped reward aggregation.

The normalizer here is indexed by **both**::

    mu[category, head]      sigma[category, head]      shape [C, H]

so e.g. ``walk/lower_body`` and ``jump/lower_body`` get independent value
normalization. The critic emits ``C * H`` normalized outputs; only the active
category's ``H`` columns are used per env.

Reuses the grouping / per-head-GAE machinery and rollout storage from
``bones_popart.py`` verbatim — this module only adds the category dimension to
the normalizer, the critic head, and the PPO update.

Key design choices (matching the implementation plan):
- Per-head GAE in *unnormalized* value space, then PopArt-normalized critic
  targets per ``(category, head)``.
- Output-preserving ART correction applied per ``(category, head)`` so the
  unnormalized critic prediction is continuous across each stats shift; only
  categories that received samples this rollout are touched.
- Default actor-advantage aggregation is ``raw`` (sum of raw per-head
  advantages × actor weights) — whitening over-amplifies low-variance penalty
  heads (prior finding), so it is opt-in.
"""

from __future__ import annotations

import json
import os
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import wandb
from rsl_rl.env import VecEnv
from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from whole_body_tracking.utils.bones_popart import (
    PER_HEAD_REWARDS_KEY,
    PER_TERM_REWARDS_RAW_KEY,
    REWARD_TERM_NAMES_KEY,
    REWARD_WEIGHTS_KEY,
    VALID_GROUPED_ACTOR_WEIGHT_MODES,
    WEIGHTED_STEP_REWARDS_KEY,
    BonesRolloutStorage,
    _build_mlp_layers,
    build_grouped_head_rewards,
    compute_grouped_actor_weights,
    compute_multi_head_gae,
    resolve_popart_groups,
    validate_popart_head_mode,
    whiten_advantages_per_head,
)

VALID_ACTOR_ADVANTAGE_SCALINGS = ("raw", "whitened", "sigma_rescaled")
DEFAULT_CATEGORY_OBS_GROUP = "category"


class DiagonalCategoryRewardPopArt(nn.Module):
    """PopArt normalizer with statistics indexed by ``(category, head)``.

    Wraps a critic ``Linear(_, C * H)`` head. Output row ``c * H + h`` predicts
    the normalized value of reward head ``h`` under motion category ``c``. The
    ART correction keeps the *unnormalized* prediction of each ``(c, h)`` output
    fixed when its stats shift.
    """

    def __init__(
        self,
        num_categories: int,
        num_heads: int,
        weight: nn.Parameter,
        bias: nn.Parameter,
        momentum: float = 0.1,
        epsilon: float = 1.0e-5,
        min_samples: int = 2,
    ):
        super().__init__()
        self.C = num_categories
        self.H = num_heads
        self.weight = weight  # reference to critic head weight, shape [C*H, in]
        self.bias = bias      # reference to critic head bias,   shape [C*H]
        self.momentum = momentum
        self.epsilon = epsilon
        self.min_samples = min_samples

        self.register_buffer("mean", torch.zeros(num_categories, num_heads))
        self.register_buffer("mean_sq", torch.full((num_categories, num_heads), epsilon))
        # Per-category debias factor: categories are seen at different rates, so
        # the EMA debiasing must be tracked separately per category.
        self.register_buffer("debias", torch.zeros(num_categories))

    # ── debiased statistics ──────────────────────────────────────────────
    def debiased_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = self.debias.clamp_min(self.epsilon).unsqueeze(-1)  # [C,1]
        mean = self.mean / debias
        mean_sq = self.mean_sq / debias
        var = (mean_sq - mean.square()).clamp_min(self.epsilon)
        return mean, var  # [C,H], [C,H]

    def debiased_std(self) -> torch.Tensor:
        _, var = self.debiased_mean_var()
        return var.sqrt()

    def mean_std(self) -> tuple[torch.Tensor, torch.Tensor]:
        mean, var = self.debiased_mean_var()
        return mean, var.sqrt()

    # ── forward (normalize / unnormalize active category) ────────────────
    def forward(self, value: torch.Tensor, category_idx: torch.Tensor, unnorm: bool = False) -> torch.Tensor:
        """value: [N, H] for the active category of each env. category_idx: [N]."""
        mean, var = self.debiased_mean_var()
        std = var.sqrt()
        m = mean[category_idx]  # [N,H]
        s = std[category_idx]   # [N,H]
        if unnorm:
            return value * s + m
        return (value - m) / s

    # ── masked update + per-(category,head) ART correction ───────────────
    @torch.no_grad()
    def update_masked(self, returns_heads: torch.Tensor, category_idx: torch.Tensor):
        """Update stats from a rollout of returns.

        Args:
            returns_heads: unnormalized per-head returns, shape ``[..., H]``.
            category_idx:  per-sample category index, broadcastable to the
                leading dims of ``returns_heads`` (e.g. ``[T, N]``).
        """
        returns_flat = returns_heads.reshape(-1, self.H).to(self.mean.dtype)
        cat_flat = category_idx.reshape(-1).long()
        device = self.mean.device

        old_mean, old_var = self.debiased_mean_var()
        old_std = old_var.sqrt()

        batch_mean = torch.zeros(self.C, self.H, dtype=self.mean.dtype, device=device)
        batch_mean_sq = torch.zeros(self.C, self.H, dtype=self.mean.dtype, device=device)
        has_data = torch.zeros(self.C, dtype=torch.bool, device=device)

        for c in range(self.C):
            mask = cat_flat == c
            if int(mask.sum()) >= self.min_samples:
                x = returns_flat[mask]  # [n_c, H]
                batch_mean[c] = x.mean(dim=0)
                batch_mean_sq[c] = x.square().mean(dim=0)
                has_data[c] = True

        if not bool(has_data.any()):
            return

        mom = self.momentum
        new_mean = self.mean.clone()
        new_mean_sq = self.mean_sq.clone()
        new_mean[has_data] = self.mean[has_data] * (1 - mom) + batch_mean[has_data] * mom
        new_mean_sq[has_data] = self.mean_sq[has_data] * (1 - mom) + batch_mean_sq[has_data] * mom
        self.mean.copy_(new_mean)
        self.mean_sq.copy_(new_mean_sq)
        self.debias[has_data] = self.debias[has_data] * (1 - mom) + mom

        new_mean_d, new_var_d = self.debiased_mean_var()
        new_std = new_var_d.sqrt()

        # Output-preserving ART correction, applied only to categories with data.
        scale = (old_std / new_std)  # [C,H]
        weight = self.weight.data.view(self.C, self.H, -1)
        bias = self.bias.data.view(self.C, self.H)
        new_bias = (old_std * bias + old_mean - new_mean_d) / new_std  # [C,H]
        rows = has_data
        bias[rows] = new_bias[rows]
        weight[rows] = weight[rows] * scale[rows].unsqueeze(-1)


class BonesCategoryRewardActorCritic(nn.Module):
    """Actor-critic whose critic emits ``C * H`` PopArt-normalized values.

    The active category is read per-env from ``obs[category_obs_group]`` and
    used to gather the relevant ``H`` columns. The category group is *not* part
    of the critic MLP input (it stays out of ``obs_groups['critic']``)."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        num_categories: int,
        num_heads: int,
        category_obs_group: str = DEFAULT_CATEGORY_OBS_GROUP,
        category_names: list[str] | None = None,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        activation: str = "swish",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        popart_momentum: float = 0.1,
        popart_epsilon: float = 1.0e-5,
        popart_min_samples: int = 2,
        **kwargs,
    ):
        if kwargs:
            print(
                "BonesCategoryRewardActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        if category_obs_group not in obs:
            raise ValueError(
                f"category_obs_group {category_obs_group!r} not present in obs groups {list(obs.keys())}. "
                "Add a 'category' observation group exposing the per-env category index."
            )

        self.obs_groups = obs_groups
        self.num_categories = num_categories
        self.num_heads = num_heads
        self.value_dim = num_categories * num_heads
        self.category_obs_group = category_obs_group
        self.category_names = (
            list(category_names) if category_names is not None else [f"cat_{i}" for i in range(num_categories)]
        )

        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])

        self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else torch.nn.Identity()
        )
        print(f"Actor MLP: {self.actor}")

        critic_layers = _build_mlp_layers(num_critic_obs, list(critic_hidden_dims), activation)
        self.critic_trunk = nn.Sequential(*critic_layers)
        self.critic_head = nn.Linear(list(critic_hidden_dims)[-1], self.value_dim)
        nn.init.uniform_(self.critic_head.weight, -1.0e-4, 1.0e-4)
        nn.init.zeros_(self.critic_head.bias)
        print(
            f"[BonesCategoryRewardActorCritic] critic head Linear(_, {self.value_dim}) "
            f"= {num_categories} categories x {num_heads} heads"
        )

        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else torch.nn.Identity()
        )
        self.value_normalizer = DiagonalCategoryRewardPopArt(
            num_categories=num_categories,
            num_heads=num_heads,
            weight=self.critic_head.weight,
            bias=self.critic_head.bias,
            momentum=popart_momentum,
            epsilon=popart_epsilon,
            min_samples=popart_min_samples,
        )

        self.noise_std_type = noise_std_type
        if noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {noise_std_type}.")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        return None

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def get_actor_obs(self, obs):
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[group] for group in self.obs_groups["critic"]], dim=-1)

    def get_category_idx(self, obs) -> torch.Tensor:
        cat = obs[self.category_obs_group]
        if cat.dim() == 2:
            cat = cat.squeeze(-1)
        return cat.long()

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=0.3)
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_obs)

    def evaluate(self, obs, unnorm: bool = False, **kwargs) -> torch.Tensor:
        """Normalized (or unnormalized) per-head values for the active category, [N, H]."""
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        all_norm = self.critic_head(self.critic_trunk(critic_obs))  # [N, C*H]
        all_norm = all_norm.view(-1, self.num_categories, self.num_heads)  # [N, C, H]
        cat = self.get_category_idx(obs)  # [N]
        active_norm = all_norm[torch.arange(all_norm.shape[0], device=all_norm.device), cat]  # [N, H]
        if unnorm:
            return self.value_normalizer(active_norm, cat, unnorm=True)
        return active_norm

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))


class BonesCategoryRewardPPO:
    policy: BonesCategoryRewardActorCritic

    def __init__(
        self,
        policy,
        reward_weights: torch.Tensor,
        head_names: list[str],
        category_names: list[str],
        head_mode: str = "grouped",
        per_head_rewards_key: str = PER_HEAD_REWARDS_KEY,
        group_membership: torch.Tensor | None = None,
        category_obs_group: str = DEFAULT_CATEGORY_OBS_GROUP,
        category_head_weights: torch.Tensor | None = None,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        learning_rate=0.001,
        max_grad_norm=1.0,
        use_clipped_value_loss=False,
        schedule="adaptive",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        popart_actor_advantage_scaling: str = "raw",
        category_adv_scaling: bool = False,
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "BonesCategoryRewardPPO.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        self.policy = policy
        self.policy.to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.storage: BonesRolloutStorage | None = None
        self.transition = BonesRolloutStorage.Transition()

        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = False
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.rnd = None
        self.symmetry = None

        validate_popart_head_mode(head_mode)
        self.head_mode = head_mode
        self.head_names = list(head_names)
        self.category_names = list(category_names)
        self.num_heads = len(self.head_names)
        self.num_categories = len(self.category_names)
        self.per_head_rewards_key = per_head_rewards_key
        self.group_membership = group_membership.to(device) if group_membership is not None else None
        self.reward_weights = reward_weights.to(device)
        self.category_head_weights = category_head_weights.to(device) if category_head_weights is not None else None
        self.category_obs_group = category_obs_group
        if popart_actor_advantage_scaling not in VALID_ACTOR_ADVANTAGE_SCALINGS:
            raise ValueError(
                f"Unsupported popart_actor_advantage_scaling={popart_actor_advantage_scaling!r}. "
                f"Expected one of {VALID_ACTOR_ADVANTAGE_SCALINGS}."
            )
        self.actor_advantage_scaling = popart_actor_advantage_scaling
        self.category_adv_scaling = category_adv_scaling
        self.last_diagnostics: dict[str, float] = {}

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != "rl":
            raise ValueError("BonesCategoryRewardPPO only supports RL training.")
        self.storage = BonesRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            value_dim=self.num_heads,
            device=self.device,
        )

    def act(self, obs):
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()  # normalized active [N,H]
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        if self.storage is None:
            raise RuntimeError("Rollout storage must be initialized before collecting transitions.")
        if self.head_mode == "grouped":
            if self.group_membership is None:
                raise RuntimeError("Grouped PopArt head mode requires a group_membership tensor.")
            if WEIGHTED_STEP_REWARDS_KEY not in extras:
                raise RuntimeError(
                    "Grouped PopArt enabled but env does not expose weighted per-term rewards under "
                    f"extras['{WEIGHTED_STEP_REWARDS_KEY}']."
                )
            per_head_rewards = build_grouped_head_rewards(
                extras[WEIGHTED_STEP_REWARDS_KEY].to(self.device),
                self.group_membership,
                self.num_heads,
            )
        elif self.per_head_rewards_key in extras:
            per_head_rewards = extras[self.per_head_rewards_key]
        else:
            raise RuntimeError(
                f"PopArt enabled but env does not expose per-head rewards under extras['{self.per_head_rewards_key}']."
            )

        self.policy.update_normalization(obs)
        self.transition.rewards = per_head_rewards.to(self.device).clone()
        self.transition.dones = dones
        time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        self.transition.time_outs = time_outs.to(self.device).clone()

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    @torch.no_grad()
    def _compute_category_adv_scale(
        self, cat_buf: torch.Tensor, advantages_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute per-category advantage scale factors for category-conditional scaling.

        For each category c, computes the fraction of samples with a positive actor
        advantage, then returns scale = 0.5 / pos_frac so every category contributes
        equally regardless of critic calibration state.

        Args:
            cat_buf: [T, N] per-step category indices.
            advantages_raw: [T, N, H] unnormalized per-head advantages.

        Returns:
            scale: [C] — multiplicative factor per category, clamped to [0.1, 5.0].
            pos_adv_frac: [C] — raw positive-advantage fraction per category.
        """
        # Scalar actor advantage per sample [T, N]
        if self.category_head_weights is not None:
            per_sample_w = self.reward_weights * self.category_head_weights[cat_buf]  # [T, N, H]
        else:
            per_sample_w = self.reward_weights  # [H] broadcast
        actor_adv = (advantages_raw * per_sample_w).sum(-1)
        cat_flat = cat_buf.reshape(-1)
        actor_adv_flat = actor_adv.reshape(-1)

        # Default to 0.5 (scale=1) for categories with no data.
        pos_adv_frac = torch.full((self.num_categories,), 0.5, device=self.device)
        for c in range(self.num_categories):
            mask = cat_flat == c
            if int(mask.sum()) >= 2:
                pos_adv_frac[c] = (actor_adv_flat[mask] > 0).float().mean()

        scale = (0.5 / pos_adv_frac.clamp_min(0.1)).clamp_max(5.0)
        return scale, pos_adv_frac

    def _category_buffer(self) -> torch.Tensor:
        """Per-step category index from stored observations, shape [T, N]."""
        cat = self.storage.observations[self.category_obs_group]  # [T, N, 1]
        return cat.reshape(cat.shape[0], cat.shape[1]).long()

    def compute_returns(self, obs):
        if self.storage is None:
            raise RuntimeError("Rollout storage must be initialized before computing returns.")

        cat_buf = self._category_buffer()  # [T, N]
        T, N = cat_buf.shape

        last_values_norm = self.policy.evaluate(obs).detach()  # [N, H]
        last_cat = self.policy.get_category_idx(obs)  # [N]

        values_norm = self.storage.values  # [T, N, H]
        values_unnorm = self.policy.value_normalizer(
            values_norm.reshape(-1, self.num_heads), cat_buf.reshape(-1), unnorm=True
        ).view(T, N, self.num_heads)
        last_values_unnorm = self.policy.value_normalizer(last_values_norm, last_cat, unnorm=True)

        returns_unnorm, advantages_raw = compute_multi_head_gae(
            rewards=self.storage.rewards,
            values=values_unnorm,
            dones=self.storage.dones.float(),
            time_outs=self.storage.time_outs.float(),
            last_values=last_values_unnorm,
            gamma=self.gamma,
            lam=self.lam,
        )

        # Category-conditional advantage scaling: re-center per-category gradient
        # contribution so every category pushes the policy with equal strength.
        cat_scale = pos_adv_frac = None
        if self.category_adv_scaling:
            cat_scale, pos_adv_frac = self._compute_category_adv_scale(cat_buf, advantages_raw)
            advantages_raw = advantages_raw * cat_scale[cat_buf].unsqueeze(-1)

        whitened_advantages, _, raw_std = whiten_advantages_per_head(advantages_raw)

        # Update PopArt stats (ART correction keeps unnorm predictions continuous),
        # then normalize the critic targets with the *post-update* stats.
        self.policy.value_normalizer.update_masked(returns_unnorm, cat_buf)
        returns_norm = self.policy.value_normalizer(
            returns_unnorm.reshape(-1, self.num_heads), cat_buf.reshape(-1)
        ).view(T, N, self.num_heads)

        self.storage.returns.copy_(returns_norm)
        self.storage.raw_advantages.copy_(advantages_raw)
        self.storage.advantages.copy_(whitened_advantages)

        self._build_diagnostics(cat_buf, returns_unnorm, advantages_raw, raw_std, pos_adv_frac, cat_scale)

    @torch.no_grad()
    def _build_diagnostics(
        self,
        cat_buf,
        returns_unnorm,
        advantages_raw,
        raw_std,
        pos_adv_frac: torch.Tensor | None = None,
        adv_scale: torch.Tensor | None = None,
    ):
        mean_ch, std_ch = self.policy.value_normalizer.mean_std()  # [C,H]
        cat_flat = cat_buf.reshape(-1)
        ret_flat = returns_unnorm.reshape(-1, self.num_heads)
        total = max(1, cat_flat.numel())
        diag: dict[str, float] = {}
        for c, cat_name in enumerate(self.category_names):
            mask = cat_flat == c
            n_c = int(mask.sum())
            diag[f"popart/fraction/{cat_name}"] = n_c / total
            if pos_adv_frac is not None:
                diag[f"popart/pos_adv_frac/{cat_name}"] = float(pos_adv_frac[c])
            if adv_scale is not None:
                diag[f"popart/adv_scale/{cat_name}"] = float(adv_scale[c])
            for h, head_name in enumerate(self.head_names):
                key = f"{cat_name}/{head_name}"
                diag[f"popart/mu/{key}"] = float(mean_ch[c, h])
                diag[f"popart/sigma/{key}"] = float(std_ch[c, h])
                if n_c >= 2:
                    rc = ret_flat[mask, h]
                    diag[f"popart/return_mean/{key}"] = float(rc.mean())
                    diag[f"popart/return_std/{key}"] = float(rc.std())
        self.last_diagnostics = diag

    def update(self):
        if self.storage is None:
            raise RuntimeError("Rollout storage must be initialized before PPO updates.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        per_head_value_loss = torch.zeros(self.num_heads, device=self.device)
        actor_adv_abs_sum = torch.zeros(self.num_heads, device=self.device)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            raw_advantages_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            _hid_states_batch,
            _masks_batch,
        ) in generator:
            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_norm_active = self.policy.evaluate(obs_batch)  # [B, H] (normalized active category)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1.0e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1.0e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Per-sample effective head weights: base reward_weights optionally
            # multiplied by per-category overrides [B, H].
            cat_ids_batch = obs_batch[self.category_obs_group].long().squeeze(-1)  # [B]
            if self.category_head_weights is not None:
                per_sample_weights = self.reward_weights * self.category_head_weights[cat_ids_batch]  # [B, H]
            else:
                per_sample_weights = self.reward_weights  # [H] broadcasts over B

            # Actor advantage aggregation across heads.
            if self.actor_advantage_scaling == "raw":
                head_adv = raw_advantages_batch * per_sample_weights
            elif self.actor_advantage_scaling == "whitened":
                head_adv = advantages_batch * per_sample_weights
            elif self.actor_advantage_scaling == "sigma_rescaled":
                sigma = self.policy.value_normalizer.debiased_std().mean(dim=0).detach()  # [H]
                head_adv = advantages_batch * sigma * per_sample_weights
            else:
                raise ValueError(f"Unsupported actor_advantage_scaling={self.actor_advantage_scaling!r}.")
            actor_adv_abs_sum += head_adv.detach().abs().reshape(-1, self.num_heads).sum(dim=0)
            actor_advantage = head_adv.sum(dim=-1)  # [B]

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -actor_advantage * ratio
            surrogate_clipped = -actor_advantage * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            value_losses = (value_norm_active - returns_batch).pow(2)
            value_loss_per_head = value_losses.mean(dim=0)
            value_loss = value_loss_per_head.mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            per_head_value_loss += value_loss_per_head.detach()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        per_head_value_loss /= num_updates
        actor_adv_share = actor_adv_abs_sum / actor_adv_abs_sum.sum().clamp_min(1.0e-8)
        self.storage.clear()

        self.last_diagnostics.update(
            {f"loss/value_per_head/{name}": per_head_value_loss[i].item() for i, name in enumerate(self.head_names)}
        )
        self.last_diagnostics.update(
            {f"popart/actor_adv_share/{name}": actor_adv_share[i].item() for i, name in enumerate(self.head_names)}
        )

        return {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

    def broadcast_parameters(self):
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self):
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel


class BonesCategoryRewardOnPolicyRunner(OnPolicyRunner):
    """Runner wiring the hierarchical (category × head) PopArt actor-critic + PPO."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None):
        deprecated_model_keys = [
            "stochastic", "init_noise_std", "noise_std_type", "state_dependent_std",
            "obs_normalization", "actor_obs_normalization", "critic_obs_normalization",
        ]
        for section in ["actor", "critic"]:
            if section in train_cfg:
                for key in deprecated_model_keys:
                    train_cfg[section].pop(key, None)
        self.registry_name = registry_name
        super().__init__(env, train_cfg, log_dir, device)

    def _resolve_categories(self) -> tuple[int, list[str]]:
        try:
            cmd = self.env.unwrapped.command_manager.get_term("motion")
        except Exception:
            cmd = None
        num_categories = getattr(cmd, "num_categories", None)
        category_names = getattr(cmd, "category_names", None)
        if num_categories is None:
            num_categories = self.alg_cfg.get("popart_num_categories")
        if num_categories is None:
            raise ValueError(
                "Hierarchical PopArt requires num_categories on the motion command term "
                "(set a categorizer on the command cfg) or alg_cfg.popart_num_categories."
            )
        num_categories = int(num_categories)
        if category_names is None:
            category_names = [f"cat_{i}" for i in range(num_categories)]
        return num_categories, list(category_names)

    def _construct_algorithm(self, obs) -> BonesCategoryRewardPPO:  # type: ignore[override]
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "`empirical_normalization` is deprecated; set actor_obs_normalization / "
                "critic_obs_normalization on the policy cfg instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        if not self.alg_cfg.get("popart_hierarchical", False):
            raise ValueError("BonesCategoryRewardOnPolicyRunner requires popart_hierarchical=True.")

        extras = getattr(self.env.unwrapped, "extras", {})
        for key in (REWARD_TERM_NAMES_KEY, REWARD_WEIGHTS_KEY, WEIGHTED_STEP_REWARDS_KEY):
            if key not in extras:
                raise RuntimeError(
                    f"Hierarchical PopArt requires env extras['{key}']. Install the bones per-term "
                    "reward manager before constructing the runner."
                )

        reward_term_names = list(extras[REWARD_TERM_NAMES_KEY])
        reward_weights = torch.as_tensor(extras[REWARD_WEIGHTS_KEY], dtype=torch.float, device=self.device)

        num_categories, category_names = self._resolve_categories()
        category_obs_group = self.alg_cfg.get("popart_category_obs_group", DEFAULT_CATEGORY_OBS_GROUP)

        policy_cfg = dict(self.policy_cfg)
        algorithm_cfg = dict(self.alg_cfg)
        for key in ("class_name",):
            policy_cfg.pop(key, None)
            algorithm_cfg.pop(key, None)
        algorithm_cfg.pop("popart_hierarchical", None)
        algorithm_cfg.pop("popart_num_categories", None)
        algorithm_cfg.pop("popart_category_obs_group", None)
        # Strip the category-only / unsupported popart knobs that don't apply here.
        for key in ("use_popart_multihead", "popart_head_mode", "popart_groups", "popart_normalize_actor_weights"):
            algorithm_cfg.pop(key, None)
        head_mode = self.alg_cfg.get("popart_head_mode", "grouped")
        validate_popart_head_mode(head_mode)
        grouped_actor_weight_mode = algorithm_cfg.pop("popart_grouped_actor_weight_mode", "uniform")
        group_preset = algorithm_cfg.pop("popart_group_preset", "upper_lower")
        popart_momentum = algorithm_cfg.pop("popart_momentum", 0.1)
        popart_epsilon = algorithm_cfg.pop("popart_epsilon", 1.0e-5)
        popart_min_samples = algorithm_cfg.pop("popart_min_samples", 2)
        category_adv_scaling = algorithm_cfg.pop("category_adv_scaling", False)
        category_head_weights_dict = algorithm_cfg.pop("popart_category_head_weights", None)

        group_membership = None
        popart_groups = None
        per_head_rewards_key = PER_TERM_REWARDS_RAW_KEY
        head_names = reward_term_names
        actor_reward_weights = reward_weights
        if head_mode == "grouped":
            if grouped_actor_weight_mode not in VALID_GROUPED_ACTOR_WEIGHT_MODES:
                raise ValueError(
                    f"Unsupported popart_grouped_actor_weight_mode={grouped_actor_weight_mode!r}."
                )
            group_names, popart_groups, group_membership = resolve_popart_groups(
                reward_term_names, self.alg_cfg.get("popart_groups"), group_preset
            )
            head_names = group_names
            actor_reward_weights = compute_grouped_actor_weights(
                reward_weights, group_membership.to(self.device), len(group_names), grouped_actor_weight_mode
            )
            weighted_step_rewards = torch.as_tensor(extras[WEIGHTED_STEP_REWARDS_KEY], device=self.device)
            extras[PER_HEAD_REWARDS_KEY] = build_grouped_head_rewards(
                weighted_step_rewards, group_membership.to(self.device), len(group_names)
            )
            per_head_rewards_key = PER_HEAD_REWARDS_KEY
        num_heads = len(head_names)

        # Build [C, H] category-head weight multiplier tensor (default: all ones).
        category_head_weights = torch.ones(num_categories, num_heads, device=self.device)
        if category_head_weights_dict:
            for cat_name, weights in category_head_weights_dict.items():
                if cat_name not in category_names:
                    print(
                        f"[WARN] popart_category_head_weights: unknown category {cat_name!r} "
                        f"(known: {category_names}), skipping"
                    )
                    continue
                if len(weights) != num_heads:
                    raise ValueError(
                        f"popart_category_head_weights[{cat_name!r}]: expected {num_heads} weights "
                        f"(heads: {head_names}), got {len(weights)}"
                    )
                c_idx = category_names.index(cat_name)
                category_head_weights[c_idx] = torch.tensor(weights, dtype=torch.float, device=self.device)
        # Use None when all-ones to skip the per-sample indexing in update().
        if category_head_weights_dict:
            _use_category_head_weights = category_head_weights
        else:
            _use_category_head_weights = None

        actor_critic = BonesCategoryRewardActorCritic(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            num_actions=self.env.num_actions,
            num_categories=num_categories,
            num_heads=num_heads,
            category_obs_group=category_obs_group,
            category_names=category_names,
            popart_momentum=popart_momentum,
            popart_epsilon=popart_epsilon,
            popart_min_samples=popart_min_samples,
            **policy_cfg,
        ).to(self.device)

        alg = BonesCategoryRewardPPO(
            actor_critic,
            reward_weights=actor_reward_weights,
            head_names=head_names,
            category_names=category_names,
            head_mode=head_mode,
            per_head_rewards_key=per_head_rewards_key,
            group_membership=group_membership,
            category_obs_group=category_obs_group,
            category_adv_scaling=category_adv_scaling,
            category_head_weights=_use_category_head_weights,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **algorithm_cfg,
        )
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])

        self.cfg["popart_hierarchical"] = True
        self.cfg["popart_category_names"] = category_names
        self.cfg["popart_head_names"] = head_names
        self.cfg["popart_num_categories"] = num_categories
        self.cfg["popart_num_heads"] = num_heads
        self.cfg["popart_groups"] = popart_groups
        self.cfg["popart_group_preset"] = group_preset if head_mode == "grouped" else None

        print("=" * 80)
        print("[BonesCategoryRewardOnPolicyRunner] Hierarchical PopArt configuration:")
        print(f"  categories ({num_categories}): {category_names}")
        print(f"  heads ({num_heads}): {head_names}")
        print(f"  critic outputs: {num_categories * num_heads} = C x H")
        print(f"  head_mode: {head_mode}  group_preset: {group_preset}")
        print(f"  actor_advantage_scaling: {alg.actor_advantage_scaling}")
        print(f"  category_adv_scaling: {alg.category_adv_scaling}")
        print(f"  grouped_actor_weight_mode: {grouped_actor_weight_mode}")
        print(f"  popart_momentum: {popart_momentum}  min_samples: {popart_min_samples}")
        print(f"  category_obs_group: {category_obs_group!r}")
        if _use_category_head_weights is not None:
            print("  category_head_weights (multipliers on top of base reward_weights):")
            for c, cat_name in enumerate(category_names):
                w = _use_category_head_weights[c].tolist()
                w_str = "  ".join(f"{head_names[h]}={w[h]:.2f}" for h in range(num_heads))
                print(f"    [{c}] {cat_name}: {w_str}")
        print("=" * 80)

        if wandb.run is not None:
            wandb.config.update(
                {
                    "popart_hierarchical": True,
                    "popart_category_names": category_names,
                    "popart_head_names": head_names,
                    "popart_num_categories": num_categories,
                    "popart_num_heads": num_heads,
                    "popart_group_preset": group_preset if head_mode == "grouped" else None,
                    "popart_actor_advantage_scaling": alg.actor_advantage_scaling,
                    "category_adv_scaling": alg.category_adv_scaling,
                    "popart_category_head_weights": (
                        {cat: _use_category_head_weights[c].tolist()
                         for c, cat in enumerate(category_names)}
                        if _use_category_head_weights is not None else None
                    ),
                },
                allow_val_change=True,
            )
        return alg

    def load_actor_only(self, path: str) -> None:
        """Load only actor weights from a checkpoint (baseline or PopArt).

        Used for two-phase training: phase-1 checkpoint provides a warm-started
        actor; the hierarchical critic starts fresh and calibrates to the already-
        competent policy rather than racing ahead of it.
        """
        import torch
        loaded = torch.load(path, map_location=self.device)
        src = loaded.get("model_state_dict", loaded)
        actor_keys = [k for k in src if k.startswith("actor") or k == "std"]
        dst = self.alg.policy.state_dict()
        loaded_keys, skipped_keys = [], []
        for k in actor_keys:
            if k in dst and dst[k].shape == src[k].shape:
                dst[k] = src[k]
                loaded_keys.append(k)
            else:
                skipped_keys.append(k)
        self.alg.policy.load_state_dict(dst)
        print(f"[load_actor_only] loaded {len(loaded_keys)} keys from {path}")
        if skipped_keys:
            print(f"[load_actor_only] skipped (shape mismatch): {skipped_keys}")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        if self.writer is None:
            return
        if locs["it"] == locs["start_iter"]:
            self.writer.add_text(
                "popart/config",
                json.dumps(
                    {
                        "categories": self.cfg.get("popart_category_names"),
                        "heads": self.cfg.get("popart_head_names"),
                        "num_categories": self.cfg.get("popart_num_categories"),
                        "num_heads": self.cfg.get("popart_num_heads"),
                        "group_preset": self.cfg.get("popart_group_preset"),
                    },
                    indent=2,
                ),
                0,
            )
        for key, value in self.alg.last_diagnostics.items():
            self.writer.add_scalar(key, value, locs["it"])

    def save(self, path: str, infos=None):
        super().save(path, infos)
        if getattr(self, "logger_type", getattr(self, "_logger_type", None)) in ["wandb"]:
            from isaaclab_rl.rsl_rl import export_policy_as_onnx
            from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx

            cmd = self.env.unwrapped.command_manager.get_term("motion")
            is_multiclip = hasattr(cmd.cfg, "zarr_path") and cmd.cfg.zarr_path
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            policy = self.alg.policy
            normalizer = getattr(policy, "actor_obs_normalizer", None)
            if is_multiclip:
                export_policy_as_onnx(policy, normalizer=normalizer, path=policy_path, filename=filename)
            else:
                export_motion_policy_as_onnx(
                    self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
                )
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
            if self.registry_name is not None and not self.registry_name.startswith("zarr:"):
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
