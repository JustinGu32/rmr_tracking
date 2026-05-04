from __future__ import annotations

import os
import json
import warnings
from collections import Counter
from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.env import VecEnv
from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb

PER_TERM_REWARDS_RAW_KEY = "per_term_rewards_raw"
PER_HEAD_REWARDS_KEY = "per_head_rewards"
WEIGHTED_STEP_REWARDS_KEY = "weighted_step_rewards"
REWARD_WEIGHTS_KEY = "reward_weights"
REWARD_TERM_NAMES_KEY = "reward_term_names"
VALID_POPART_HEAD_MODES = ("per_term", "grouped")
VALID_GROUPED_ACTOR_WEIGHT_MODES = ("uniform", "sum_user_weights")
VALID_POPART_GROUP_PRESETS = ("upper_lower", "actual_individual")
DEFAULT_POPART_GROUPS_UPPER_LOWER = {
    "global_pose_tracking": [
        "motion_global_anchor_pos",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
    ],
    "lower_limb_tracking": [
        "vr_position_lower",
    ],
    "motion_dynamics": [
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ],
    "regularization_constraints": [
        "action_rate_l2",
        "joint_limit",
        "undesired_contacts",
    ],
    "upper_limb_tracking": [
        "vr_position_upper",
    ],
}
DEFAULT_POPART_GROUPS_ACTUAL_INDIVIDUAL = {
    "left_wrist_tracking": ["vr_position_left_wrist"],
    "right_wrist_tracking": ["vr_position_right_wrist"],
    "torso_tracking": ["vr_position_torso"],
    "left_ankle_tracking": ["vr_position_left_ankle"],
    "right_ankle_tracking": ["vr_position_right_ankle"],
    "pelvis_tracking": ["vr_position_pelvis"],
    "global_pose_tracking": [
        "motion_global_anchor_pos",
        "motion_global_anchor_ori",
        "motion_body_pos",
        "motion_body_ori",
    ],
    "motion_dynamics": [
        "motion_body_lin_vel",
        "motion_body_ang_vel",
    ],
    "regularization_constraints": [
        "action_rate_l2",
        "joint_limit",
        "undesired_contacts",
    ],
}
DEFAULT_POPART_GROUPS_BY_PRESET = {
    "upper_lower": DEFAULT_POPART_GROUPS_UPPER_LOWER,
    "actual_individual": DEFAULT_POPART_GROUPS_ACTUAL_INDIVIDUAL,
}


def validate_popart_head_mode(head_mode: str) -> None:
    if head_mode not in VALID_POPART_HEAD_MODES:
        raise ValueError(
            f"Unsupported popart_head_mode={head_mode!r}. Expected one of {VALID_POPART_HEAD_MODES}."
        )


def validate_popart_group_preset(group_preset: str) -> None:
    if group_preset not in VALID_POPART_GROUP_PRESETS:
        raise ValueError(
            f"Unsupported popart_group_preset={group_preset!r}. Expected one of {VALID_POPART_GROUP_PRESETS}."
        )


def resolve_popart_groups(
    reward_term_names: list[str],
    popart_groups: Mapping[str, list[str]] | None,
    popart_group_preset: str = "upper_lower",
) -> tuple[list[str], dict[str, list[str]], torch.Tensor]:
    validate_popart_group_preset(popart_group_preset)
    groups = DEFAULT_POPART_GROUPS_BY_PRESET[popart_group_preset] if popart_groups is None else popart_groups
    if not isinstance(groups, Mapping):
        raise ValueError("popart_groups must be a mapping from group name to a list of reward term names.")

    group_names = list(groups.keys())
    normalized_groups = {group_name: list(term_names) for group_name, term_names in groups.items()}
    assigned_terms = [term_name for term_names in normalized_groups.values() for term_name in term_names]
    assigned_counter = Counter(assigned_terms)
    active_term_set = set(reward_term_names)

    empty_groups = [group_name for group_name, term_names in normalized_groups.items() if len(term_names) == 0]
    duplicated_terms = sorted(term_name for term_name, count in assigned_counter.items() if count > 1)
    unknown_terms = sorted(term_name for term_name in assigned_counter if term_name not in active_term_set)
    unassigned_terms = sorted(term_name for term_name in reward_term_names if term_name not in assigned_counter)

    if empty_groups or duplicated_terms or unknown_terms or unassigned_terms:
        problems: list[str] = []
        if empty_groups:
            problems.append(f"empty groups: {empty_groups}")
        if duplicated_terms:
            problems.append(f"duplicated terms: {duplicated_terms}")
        if unknown_terms:
            problems.append(f"unknown terms: {unknown_terms}")
        if unassigned_terms:
            problems.append(f"unassigned terms: {unassigned_terms}")
        raise ValueError("Invalid PopArt grouped head definition: " + "; ".join(problems))

    term_to_group = {
        term_name: group_index
        for group_index, group_name in enumerate(group_names)
        for term_name in normalized_groups[group_name]
    }
    group_membership = torch.tensor([term_to_group[term_name] for term_name in reward_term_names], dtype=torch.long)
    return group_names, normalized_groups, group_membership


def build_grouped_head_rewards(
    weighted_step_reward: torch.Tensor,
    group_membership: torch.Tensor,
    num_groups: int,
) -> torch.Tensor:
    if weighted_step_reward.ndim != 2:
        raise ValueError(
            f"Expected weighted_step_reward with shape (num_envs, num_terms), received {tuple(weighted_step_reward.shape)}."
        )
    if group_membership.ndim != 1 or group_membership.shape[0] != weighted_step_reward.shape[1]:
        raise ValueError(
            "group_membership must be a 1D tensor whose length matches weighted_step_reward.shape[1]. "
            f"Received {tuple(group_membership.shape)} for rewards {tuple(weighted_step_reward.shape)}."
        )
    grouped_rewards = torch.zeros(
        (weighted_step_reward.shape[0], num_groups),
        dtype=weighted_step_reward.dtype,
        device=weighted_step_reward.device,
    )
    grouped_rewards.index_add_(1, group_membership.to(weighted_step_reward.device), weighted_step_reward)
    return grouped_rewards


def compute_grouped_actor_weights(
    reward_weights: torch.Tensor,
    group_membership: torch.Tensor,
    num_groups: int,
    grouped_actor_weight_mode: str,
) -> torch.Tensor:
    if grouped_actor_weight_mode == "uniform":
        return torch.ones(num_groups, dtype=reward_weights.dtype, device=reward_weights.device)
    if grouped_actor_weight_mode == "sum_user_weights":
        grouped_weights = torch.zeros(num_groups, dtype=reward_weights.dtype, device=reward_weights.device)
        grouped_weights.index_add_(0, group_membership.to(reward_weights.device), reward_weights)
        return grouped_weights
    raise ValueError(
        "Unsupported popart_grouped_actor_weight_mode="
        f"{grouped_actor_weight_mode!r}. Expected one of {VALID_GROUPED_ACTOR_WEIGHT_MODES}."
    )


def _build_mlp_layers(input_dim: int, hidden_dims: list[int], activation: str) -> list[nn.Module]:
    if not hidden_dims:
        raise ValueError("PopArt critic requires at least one hidden dimension.")
    layers: list[nn.Module] = []
    activation_cls = type(MLP(1, 1, [1], activation=activation)[1])
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(activation_cls())
        prev_dim = hidden_dim
    return layers


def compute_multi_head_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    time_outs: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-head GAE in unnormalized value space."""

    returns = torch.zeros_like(values)
    advantages = torch.zeros_like(values)
    advantage = torch.zeros_like(last_values)

    for step in reversed(range(rewards.shape[0])):
        if step == rewards.shape[0] - 1:
            next_values = last_values
        else:
            next_values = values[step + 1]

        done = dones[step].float()
        timeout = time_outs[step].float()
        true_terminated = done * (1.0 - timeout)
        bootstrap_mask = 1.0 - true_terminated
        recursion_mask = 1.0 - done

        delta = rewards[step] + gamma * bootstrap_mask * next_values - values[step]
        advantage = delta + gamma * lam * recursion_mask * advantage
        advantages[step] = advantage
        returns[step] = advantage + values[step]

    return returns, advantages


def whiten_advantages_per_head(advantages: torch.Tensor, eps: float = 1.0e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flat_advantages = advantages.view(-1, advantages.shape[-1])
    unbiased = flat_advantages.shape[0] > 1
    std, mean = torch.std_mean(flat_advantages, dim=0, unbiased=unbiased)
    whitened = (advantages - mean.view(1, 1, -1)) / (std.view(1, 1, -1) + eps)
    return whitened, mean, std


def compute_scalar_weighted_reward(per_term_rewards_raw: torch.Tensor, reward_weights: torch.Tensor) -> torch.Tensor:
    """Collapse per-head raw rewards back to the scalar weighted reward used by the env."""
    view_shape = [1] * (per_term_rewards_raw.ndim - 1) + [reward_weights.shape[0]]
    return (per_term_rewards_raw * reward_weights.view(*view_shape)).sum(dim=-1)


class DiagonalPopArt(nn.Module):
    """PopArt normalizer that preserves unnormalized critic outputs exactly."""

    def __init__(
        self,
        value_dim: int,
        weight: nn.Parameter,
        bias: nn.Parameter,
        momentum: float = 0.1,
        epsilon: float = 1.0e-5,
    ):
        super().__init__()
        self.value_dim = value_dim
        self.weight = weight
        self.bias = bias
        self.momentum = momentum
        self.epsilon = epsilon

        self.register_buffer("mean", torch.zeros(value_dim))
        self.register_buffer("mean_sq", torch.full((value_dim,), epsilon))
        self.register_buffer("debias", torch.zeros(1))

    def debiased_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        debias = self.debias.clamp_min(self.epsilon)
        mean = self.mean / debias
        mean_sq = self.mean_sq / debias
        var = (mean_sq - mean.square()).clamp_min(self.epsilon)
        return mean, var

    def debiased_std(self) -> torch.Tensor:
        _, var = self.debiased_mean_var()
        return torch.sqrt(var)

    def forward(self, value: torch.Tensor, unnorm: bool = False) -> torch.Tensor:
        mean, _ = self.debiased_mean_var()
        std = self.debiased_std()
        if unnorm:
            return value * std + mean
        return (value - mean) / std

    @torch.no_grad()
    def update(self, targets: torch.Tensor):
        if targets.ndim != 2 or targets.shape[-1] != self.value_dim:
            raise ValueError(
                f"Expected PopArt targets with shape [batch, {self.value_dim}], received {tuple(targets.shape)}."
            )

        old_mean, _ = self.debiased_mean_var()
        old_std = self.debiased_std()

        batch_mean = targets.mean(dim=0)
        batch_mean_sq = targets.square().mean(dim=0)

        self.mean.mul_(1.0 - self.momentum).add_(batch_mean, alpha=self.momentum)
        self.mean_sq.mul_(1.0 - self.momentum).add_(batch_mean_sq, alpha=self.momentum)
        self.debias.mul_(1.0 - self.momentum).add_(self.momentum)

        new_mean, _ = self.debiased_mean_var()
        new_std = self.debiased_std()

        scale = (old_std / new_std).unsqueeze(-1)
        self.weight.data.mul_(scale)
        self.bias.data.copy_((old_std * self.bias.data + old_mean - new_mean) / new_std)


class BonesPopArtActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        value_dim: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        critic_hidden_dims: list[int] | tuple[int, ...] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        popart_momentum: float = 0.1,
        popart_epsilon: float = 1.0e-5,
        **kwargs,
    ):
        if kwargs:
            print(
                "BonesPopArtActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.value_dim = value_dim

        num_actor_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["policy"])
        num_critic_obs = sum(obs[group_name].shape[-1] for group_name in obs_groups["critic"])

        self.actor = MLP(num_actor_obs, num_actions, list(actor_hidden_dims), activation)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = (
            EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else torch.nn.Identity()
        )
        print(f"Actor MLP: {self.actor}")

        critic_layers = _build_mlp_layers(num_critic_obs, list(critic_hidden_dims), activation)
        self.critic_trunk = nn.Sequential(*critic_layers)
        self.critic_head = nn.Linear(list(critic_hidden_dims)[-1], value_dim)
        nn.init.uniform_(self.critic_head.weight, -1.0e-4, 1.0e-4)
        nn.init.zeros_(self.critic_head.bias)

        self.critic_obs_normalization = critic_obs_normalization
        self.critic_obs_normalizer = (
            EmpiricalNormalization(num_critic_obs) if critic_obs_normalization else torch.nn.Identity()
        )
        self.value_normalizer = DiagonalPopArt(
            value_dim=value_dim,
            weight=self.critic_head.weight,
            bias=self.critic_head.bias,
            momentum=popart_momentum,
            epsilon=popart_epsilon,
        )

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

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
        return torch.cat([obs[group_name] for group_name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[group_name] for group_name in self.obs_groups["critic"]], dim=-1)

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

    def evaluate(self, obs, unnorm: bool = False, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        normalized_values = self.critic_head(self.critic_trunk(critic_obs))
        if unnorm:
            return self.value_normalizer(normalized_values, unnorm=True)
        return normalized_values

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict: bool = True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class BonesRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.time_outs = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs, num_transitions_per_env, obs, actions_shape, value_dim: int, device="cpu"):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actions_shape = actions_shape
        self.value_dim = value_dim

        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=device,
        )
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.time_outs = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=device)
        self.raw_advantages = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, value_dim, device=device)
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow. Call clear() before adding new transitions.")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.time_outs[self.step].copy_(transition.time_outs.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches, num_epochs=1):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        raw_advantages = self.raw_advantages.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for index in range(num_mini_batches):
                start = index * mini_batch_size
                end = (index + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    raw_advantages[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                )


class BonesPopArtPPO:
    policy: BonesPopArtActorCritic
    VALID_ACTOR_ADVANTAGE_SCALINGS = ("whitened", "sigma_rescaled", "raw")

    def __init__(
        self,
        policy,
        reward_weights: torch.Tensor,
        reward_term_names: list[str],
        head_mode: str = "per_term",
        per_head_rewards_key: str = PER_TERM_REWARDS_RAW_KEY,
        group_membership: torch.Tensor | None = None,
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
        popart_normalize_actor_weights: bool = False,
        popart_actor_advantage_scaling: str = "whitened",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "BonesPopArtPPO.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
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
        self.policy.to(self.device)
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
        self.head_names = list(reward_term_names)
        self.reward_term_names = self.head_names
        self.per_head_rewards_key = per_head_rewards_key
        self.group_membership = group_membership.to(self.device) if group_membership is not None else None
        self.reward_weights = reward_weights.to(self.device)
        if popart_normalize_actor_weights:
            denom = self.reward_weights.abs().sum().clamp_min(1.0e-8)
            self.reward_weights = self.reward_weights * (self.reward_weights.numel() / denom)
        self.popart_normalize_actor_weights = popart_normalize_actor_weights
        if popart_actor_advantage_scaling not in self.VALID_ACTOR_ADVANTAGE_SCALINGS:
            raise ValueError(
                "Unsupported popart_actor_advantage_scaling="
                f"{popart_actor_advantage_scaling!r}. Expected one of {self.VALID_ACTOR_ADVANTAGE_SCALINGS}."
            )
        self.actor_advantage_scaling = popart_actor_advantage_scaling

        self.last_diagnostics: dict[str, float] = {}

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != "rl":
            raise ValueError("BonesPopArtPPO only supports RL training.")
        self.storage = BonesRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            value_dim=self.reward_weights.numel(),
            device=self.device,
        )

    def act(self, obs):
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
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
                self.reward_weights.numel(),
            )
            extras[PER_HEAD_REWARDS_KEY] = per_head_rewards
            extras[PER_TERM_REWARDS_RAW_KEY] = per_head_rewards
        elif self.per_head_rewards_key in extras:
            per_head_rewards = extras[self.per_head_rewards_key]
        else:
            raise RuntimeError(
                "PopArt enabled but env does not expose per-head rewards under "
                f"extras['{self.per_head_rewards_key}']."
            )

        self.policy.update_normalization(obs)
        self.transition.rewards = per_head_rewards.to(self.device).clone()
        self.transition.dones = dones
        time_outs = extras.get("time_outs", torch.zeros_like(dones, dtype=torch.bool))
        self.transition.time_outs = time_outs.to(self.device).clone()

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        if self.storage is None:
            raise RuntimeError("Rollout storage must be initialized before computing returns.")

        last_values = self.policy.evaluate(obs).detach()
        values_unnorm = self.policy.value_normalizer(self.storage.values, unnorm=True)
        last_values_unnorm = self.policy.value_normalizer(last_values, unnorm=True)
        returns_unnorm, advantages_raw = compute_multi_head_gae(
            rewards=self.storage.rewards,
            values=values_unnorm,
            dones=self.storage.dones.float(),
            time_outs=self.storage.time_outs.float(),
            last_values=last_values_unnorm,
            gamma=self.gamma,
            lam=self.lam,
        )
        whitened_advantages, _, raw_std = whiten_advantages_per_head(advantages_raw)
        flat_returns = returns_unnorm.view(-1, returns_unnorm.shape[-1])
        preds_unnorm_for_logging = values_unnorm.view(-1, self.reward_weights.numel()).clone()
        self.policy.value_normalizer.update(flat_returns)

        self.storage.returns.copy_(self.policy.value_normalizer(returns_unnorm))
        self.storage.raw_advantages.copy_(advantages_raw)
        self.storage.advantages.copy_(whitened_advantages)

        whitened_std = whitened_advantages.view(-1, whitened_advantages.shape[-1]).std(
            dim=0, unbiased=whitened_advantages.shape[0] * whitened_advantages.shape[1] > 1
        )
        popart_mean, _ = self.policy.value_normalizer.debiased_mean_var()
        popart_std = self.policy.value_normalizer.debiased_std()
        diagnostics = {
            f"popart/mu/{head_name}": popart_mean[index].item()
            for index, head_name in enumerate(self.head_names)
        }
        diagnostics.update(
            {
                f"popart/sigma/{head_name}": popart_std[index].item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        diagnostics.update(
            {
                f"value/{head_name}/pred_mean_unnorm": preds_unnorm_for_logging[:, index].mean().item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        diagnostics.update(
            {
                f"return/{head_name}/mean_unnorm": flat_returns[:, index].mean().item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        diagnostics.update(
            {
                f"advantage/{head_name}/std_unwhitened": raw_std[index].item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        diagnostics.update(
            {
                f"advantage/{head_name}/std_whitened": whitened_std[index].item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        diagnostics.update(
            {
                f"reward_weight/{head_name}": self.reward_weights[index].item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        self.last_diagnostics = diagnostics

    def update(self):
        if self.storage is None:
            raise RuntimeError("Rollout storage must be initialized before PPO updates.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        per_head_value_loss = torch.zeros(self.reward_weights.numel(), device=self.device)
        actor_adv_share_sum = torch.zeros(self.reward_weights.numel(), device=self.device)
        actor_coefficients = self.reward_weights.detach().clone()
        if self.actor_advantage_scaling == "sigma_rescaled":
            actor_coefficients = actor_coefficients * self.policy.value_normalizer.debiased_std().detach()

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
            if self.normalize_advantage_per_mini_batch and self.actor_advantage_scaling != "raw":
                advantages_batch, _, _ = whiten_advantages_per_head(advantages_batch.view(1, -1, advantages_batch.shape[-1]))
                advantages_batch = advantages_batch.view_as(returns_batch)

            self.policy.act(obs_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch)
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

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            if self.actor_advantage_scaling == "whitened":
                actor_advantages = advantages_batch * self.reward_weights
            elif self.actor_advantage_scaling == "sigma_rescaled":
                sigma = self.policy.value_normalizer.debiased_std().detach()
                actor_advantages = advantages_batch * sigma * self.reward_weights
            elif self.actor_advantage_scaling == "raw":
                actor_advantages = raw_advantages_batch * self.reward_weights
            else:
                raise ValueError(
                    "Unsupported actor_advantage_scaling="
                    f"{self.actor_advantage_scaling!r}. Expected one of {self.VALID_ACTOR_ADVANTAGE_SCALINGS}."
                )
            actor_adv_share = actor_advantages.abs()
            actor_adv_share = actor_adv_share / actor_adv_share.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            actor_adv_share_sum += actor_adv_share.mean(dim=0)
            surrogate = ratio.unsqueeze(-1) * actor_advantages
            surrogate_clipped = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param).unsqueeze(-1) * actor_advantages
            surrogate_loss = -torch.min(surrogate, surrogate_clipped).sum(dim=-1).mean()

            value_losses = (value_batch - returns_batch).pow(2)
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
        actor_adv_share_sum /= num_updates
        self.storage.clear()

        self.last_diagnostics.update(
            {
                f"loss/value_per_head/{head_name}": per_head_value_loss[index].item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        self.last_diagnostics.update(
            {
                f"popart/actor_coeffs/{head_name}": actor_coefficients[index].item()
                for index, head_name in enumerate(self.head_names)
            }
        )
        self.last_diagnostics.update(
            {
                f"popart/adv_share/{head_name}": actor_adv_share_sum[index].item()
                for index, head_name in enumerate(self.head_names)
            }
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


class BonesPopArtOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None):
        deprecated_model_keys = [
            "stochastic",
            "init_noise_std",
            "noise_std_type",
            "state_dependent_std",
            "obs_normalization",
            "actor_obs_normalization",
            "critic_obs_normalization",
        ]
        for section in ["actor", "critic"]:
            if section in train_cfg:
                for key in deprecated_model_keys:
                    train_cfg[section].pop(key, None)
        self.registry_name = registry_name
        super().__init__(env, train_cfg, log_dir, device)

    def _construct_algorithm(self, obs) -> BonesPopArtPPO:  # type: ignore[override]
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        if not self.alg_cfg.get("use_popart_multihead", False):
            raise ValueError("BonesPopArtOnPolicyRunner requires use_popart_multihead=True.")
        head_mode = self.alg_cfg.get("popart_head_mode", "per_term")
        validate_popart_head_mode(head_mode)
        actor_advantage_scaling = self.alg_cfg.get("popart_actor_advantage_scaling", "whitened")
        if actor_advantage_scaling not in BonesPopArtPPO.VALID_ACTOR_ADVANTAGE_SCALINGS:
            raise ValueError(
                "Unsupported popart_actor_advantage_scaling="
                f"{actor_advantage_scaling!r}. Expected one of {BonesPopArtPPO.VALID_ACTOR_ADVANTAGE_SCALINGS}."
            )

        extras = getattr(self.env.unwrapped, "extras", {})
        if REWARD_TERM_NAMES_KEY not in extras or REWARD_WEIGHTS_KEY not in extras:
            raise RuntimeError(
                "PopArt enabled but env extras do not expose reward metadata. "
                "Install the bones per-term reward manager before wrapping the environment."
            )
        if PER_TERM_REWARDS_RAW_KEY not in extras:
            raise RuntimeError(
                f"PopArt enabled but env does not expose extras['{PER_TERM_REWARDS_RAW_KEY}']. "
                "Install BonesPerTermRewardManager on the env before constructing the runner."
            )

        reward_term_names = list(extras[REWARD_TERM_NAMES_KEY])
        reward_weights = torch.as_tensor(extras[REWARD_WEIGHTS_KEY], dtype=torch.float, device=self.device)
        per_term_tensor = torch.as_tensor(extras[PER_TERM_REWARDS_RAW_KEY], device=self.device)
        num_terms = len(reward_term_names)
        if per_term_tensor.ndim != 2 or per_term_tensor.shape[-1] != num_terms:
            raise RuntimeError(
                f"extras['{PER_TERM_REWARDS_RAW_KEY}'] has shape {tuple(per_term_tensor.shape)}, "
                f"expected (num_envs, {num_terms})."
            )
        if reward_weights.shape != (num_terms,):
            raise RuntimeError(
                f"extras['{REWARD_WEIGHTS_KEY}'] has shape {tuple(reward_weights.shape)}, "
                f"expected ({num_terms},)."
            )

        policy_cfg = dict(self.policy_cfg)
        algorithm_cfg = dict(self.alg_cfg)
        policy_cfg.pop("class_name", None)
        algorithm_cfg.pop("class_name", None)
        algorithm_cfg.pop("use_popart_multihead", None)
        algorithm_cfg.pop("popart_head_mode", None)
        algorithm_cfg.pop("popart_groups", None)
        grouped_actor_weight_mode = algorithm_cfg.pop("popart_grouped_actor_weight_mode", "uniform")
        group_preset = algorithm_cfg.pop("popart_group_preset", "upper_lower")
        popart_momentum = algorithm_cfg.pop("popart_momentum", 0.1)
        popart_epsilon = algorithm_cfg.pop("popart_epsilon", 1.0e-5)

        group_names: list[str] | None = None
        popart_groups: dict[str, list[str]] | None = None
        group_membership: torch.Tensor | None = None
        per_head_rewards_key = PER_TERM_REWARDS_RAW_KEY
        head_names = reward_term_names
        actor_reward_weights = reward_weights
        if head_mode == "grouped":
            if grouped_actor_weight_mode not in VALID_GROUPED_ACTOR_WEIGHT_MODES:
                raise ValueError(
                    "Unsupported popart_grouped_actor_weight_mode="
                    f"{grouped_actor_weight_mode!r}. Expected one of {VALID_GROUPED_ACTOR_WEIGHT_MODES}."
                )
            if WEIGHTED_STEP_REWARDS_KEY not in extras:
                raise RuntimeError(
                    "Grouped PopArt enabled but env does not expose weighted per-term rewards under "
                    f"extras['{WEIGHTED_STEP_REWARDS_KEY}']."
                )
            weighted_step_rewards = torch.as_tensor(extras[WEIGHTED_STEP_REWARDS_KEY], device=self.device)
            if weighted_step_rewards.ndim != 2 or weighted_step_rewards.shape[-1] != num_terms:
                raise RuntimeError(
                    f"extras['{WEIGHTED_STEP_REWARDS_KEY}'] has shape {tuple(weighted_step_rewards.shape)}, "
                    f"expected (num_envs, {num_terms})."
                )
            group_names, popart_groups, group_membership = resolve_popart_groups(
                reward_term_names,
                self.alg_cfg.get("popart_groups"),
                group_preset,
            )
            head_names = group_names
            actor_reward_weights = compute_grouped_actor_weights(
                reward_weights,
                group_membership.to(self.device),
                len(group_names),
                grouped_actor_weight_mode,
            )
            extras[PER_HEAD_REWARDS_KEY] = build_grouped_head_rewards(
                weighted_step_rewards,
                group_membership.to(self.device),
                len(group_names),
            )
            extras[PER_TERM_REWARDS_RAW_KEY] = extras[PER_HEAD_REWARDS_KEY]
            per_head_rewards_key = PER_HEAD_REWARDS_KEY
        value_dim = len(head_names)

        actor_critic = BonesPopArtActorCritic(
            obs=obs,
            obs_groups=self.cfg["obs_groups"],
            num_actions=self.env.num_actions,
            value_dim=value_dim,
            popart_momentum=popart_momentum,
            popart_epsilon=popart_epsilon,
            **policy_cfg,
        ).to(self.device)
        alg = BonesPopArtPPO(
            actor_critic,
            reward_weights=actor_reward_weights,
            reward_term_names=head_names,
            head_mode=head_mode,
            per_head_rewards_key=per_head_rewards_key,
            group_membership=group_membership,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **algorithm_cfg,
        )
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])

        self.cfg["popart_head_mode"] = head_mode
        self.cfg["popart_head_names"] = head_names
        self.cfg["popart_reward_term_names"] = reward_term_names
        self.cfg["popart_reward_weights"] = actor_reward_weights.detach().cpu().tolist()
        self.cfg["popart_value_dim"] = value_dim
        self.cfg["popart_groups"] = popart_groups
        self.cfg["popart_group_preset"] = group_preset if head_mode == "grouped" else None
        self.cfg["popart_grouped_actor_weight_mode"] = grouped_actor_weight_mode if head_mode == "grouped" else None
        self.cfg["popart_terminate_reward_injection"] = False
        if wandb.run is not None:
            wandb.config.update(
                {
                    "popart_head_mode": head_mode,
                    "popart_head_names": head_names,
                    "popart_groups": popart_groups,
                    "popart_group_preset": group_preset if head_mode == "grouped" else None,
                    "popart_grouped_actor_weight_mode": (
                        grouped_actor_weight_mode if head_mode == "grouped" else None
                    ),
                    "popart_reward_weights": actor_reward_weights.detach().cpu().tolist(),
                    "popart_value_dim": value_dim,
                },
                allow_val_change=True,
            )

        print("=" * 80)
        print("[BonesPopArtOnPolicyRunner] PopArt configuration:")
        print(f"  value_dim: {value_dim}")
        print(f"  head_mode: {head_mode}")
        if head_mode == "grouped":
            print("  grouped head definitions:")
            for index, group_name in enumerate(head_names):
                print(
                    f"    [{index:2d}] {group_name:28s} terms={popart_groups[group_name]} "
                    f"actor_weight={actor_reward_weights[index].item():+.4f}"
                )
            print(f"  grouped_preset: {group_preset}")
            print(f"  grouped_actor_weight_mode: {grouped_actor_weight_mode}")
        else:
            print("  reward term names (in head order):")
            for index, name in enumerate(head_names):
                print(f"    [{index:2d}] {name:40s} weight={actor_reward_weights[index].item():+.4f}")
        print(f"  popart_normalize_actor_weights: {alg.popart_normalize_actor_weights}")
        print(f"  popart_actor_advantage_scaling: {alg.actor_advantage_scaling}")
        print(f"  popart_momentum: {popart_momentum}")
        print(f"  popart_epsilon: {popart_epsilon}")
        print("  terminate_reward_injection: False (v1 deviation from CompositeMotion)")
        print("=" * 80)
        return alg

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        if self.writer is None:
            return
        if locs["it"] == locs["start_iter"]:
            self.writer.add_text(
                "popart/config",
                json.dumps(
                    {
                        "head_mode": self.cfg["popart_head_mode"],
                        "head_names": self.cfg["popart_head_names"],
                        "value_dim": self.cfg["popart_value_dim"],
                        "reward_term_names": self.cfg["popart_reward_term_names"],
                        "groups": self.cfg["popart_groups"],
                        "group_preset": self.cfg["popart_group_preset"],
                        "grouped_actor_weight_mode": self.cfg["popart_grouped_actor_weight_mode"],
                        "reward_weights": self.cfg["popart_reward_weights"],
                        "normalize_actor_weights": self.alg.popart_normalize_actor_weights,
                        "actor_advantage_scaling": self.alg.actor_advantage_scaling,
                        "terminate_reward_injection": False,
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
