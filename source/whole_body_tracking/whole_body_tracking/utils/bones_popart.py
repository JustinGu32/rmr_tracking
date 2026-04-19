from __future__ import annotations

import os
import statistics
import time
import warnings
from collections import deque

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.runners.on_policy_runner import OnPolicyRunner
from rsl_rl.storage.rollout_storage import RolloutStorage
from rsl_rl.utils import resolve_obs_groups

DEFAULT_REWARD_HEADS = ["upper", "lower", "global"]
BALANCED_REWARD_HEADS = [
    "limb_tracking",
    "global_pose_tracking",
    "motion_dynamics",
    "regularization_constraints",
]
INDIVIDUAL_REWARD_HEADS = ["left_arm", "right_arm", "torso", "left_leg", "right_leg", "pelvis", "global"]

BALANCED_REWARD_HEAD_TERMS = {
    "limb_tracking": [
        "vr_position_left_arm",
        "vr_position_right_arm",
        "vr_position_torso",
        "vr_position_left_leg",
        "vr_position_right_leg",
        "vr_position_pelvis",
    ],
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


def should_use_bones_popart_runner(train_cfg: dict | object) -> bool:
    cfg = _get_bones_popart_cfg(train_cfg)
    return bool(cfg.get("enabled", False))


def _get_bones_popart_cfg(train_cfg: dict | object) -> dict:
    if isinstance(train_cfg, dict):
        cfg = train_cfg.get("bones_popart", {})
    else:
        cfg = getattr(train_cfg, "bones_popart", {})
    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise TypeError(f"Expected bones_popart config to be a dict, got: {type(cfg)}")
    return cfg


def _compute_reward_head_vector(
    step_reward: torch.Tensor,
    term_names: list[str],
    dt: float,
    reward_heads: list[str] | None = None,
) -> tuple[torch.Tensor, list[str]]:
    reward_heads = reward_heads or DEFAULT_REWARD_HEADS

    if reward_heads == DEFAULT_REWARD_HEADS:
        try:
            upper_idx = term_names.index("vr_position_upper")
            lower_idx = term_names.index("vr_position_lower")
        except ValueError as exc:
            raise RuntimeError(
                "Bones reward vectors require compliance reward terms 'vr_position_upper' and 'vr_position_lower'."
            ) from exc

        head_reward = torch.zeros(step_reward.shape[0], 3, dtype=step_reward.dtype, device=step_reward.device)
        head_reward[:, 0] = step_reward[:, upper_idx]
        head_reward[:, 1] = step_reward[:, lower_idx]

        global_mask = torch.ones(len(term_names), dtype=torch.bool, device=step_reward.device)
        global_mask[upper_idx] = False
        global_mask[lower_idx] = False
        head_reward[:, 2] = step_reward[:, global_mask].sum(dim=-1)

        return head_reward, reward_heads

    elif reward_heads == BALANCED_REWARD_HEADS:
        term_index = {name: idx for idx, name in enumerate(term_names)}
        missing_terms = [
            term_name
            for head_name in BALANCED_REWARD_HEADS
            for term_name in BALANCED_REWARD_HEAD_TERMS[head_name]
            if term_name not in term_index
        ]
        if missing_terms:
            missing = ", ".join(missing_terms)
            raise RuntimeError(f"Balanced bones reward vectors require reward terms: {missing}")

        head_reward = torch.zeros(
            step_reward.shape[0], len(BALANCED_REWARD_HEADS), dtype=step_reward.dtype, device=step_reward.device
        )
        for head_idx, head_name in enumerate(BALANCED_REWARD_HEADS):
            indices = [term_index[term_name] for term_name in BALANCED_REWARD_HEAD_TERMS[head_name]]
            head_reward[:, head_idx] = step_reward[:, indices].sum(dim=-1)

        return head_reward, reward_heads

    elif reward_heads == INDIVIDUAL_REWARD_HEADS:
        try:
            l_arm_idx = term_names.index("vr_position_left_arm")
            r_arm_idx = term_names.index("vr_position_right_arm")
            torso_idx = term_names.index("vr_position_torso")
            l_leg_idx = term_names.index("vr_position_left_leg")
            r_leg_idx = term_names.index("vr_position_right_leg")
            pelvis_idx = term_names.index("vr_position_pelvis")
        except ValueError as exc:
            raise RuntimeError(
                "Individual bones reward vectors require specific limb terms (vr_position_left_arm, etc.)."
            ) from exc

        head_reward = torch.zeros(step_reward.shape[0], 7, dtype=step_reward.dtype, device=step_reward.device)
        head_reward[:, 0] = step_reward[:, l_arm_idx]
        head_reward[:, 1] = step_reward[:, r_arm_idx]
        head_reward[:, 2] = step_reward[:, torso_idx]
        head_reward[:, 3] = step_reward[:, l_leg_idx]
        head_reward[:, 4] = step_reward[:, r_leg_idx]
        head_reward[:, 5] = step_reward[:, pelvis_idx]

        global_mask = torch.ones(len(term_names), dtype=torch.bool, device=step_reward.device)
        for idx in [l_arm_idx, r_arm_idx, torso_idx, l_leg_idx, r_leg_idx, pelvis_idx]:
            global_mask[idx] = False
        head_reward[:, 6] = step_reward[:, global_mask].sum(dim=-1)

        return head_reward, reward_heads

    else:
        raise ValueError(
            f"Only the default, balanced, or individual reward heads are supported in v1, got: {reward_heads}"
        )


class BonesRewardVectorWrapper(gym.Wrapper):
    """Adds a learning-only reward vector while preserving the scalar env reward path."""

    def __init__(self, env: gym.Env, reward_heads: list[str] | None = None):
        super().__init__(env)
        self.reward_heads = reward_heads or DEFAULT_REWARD_HEADS
        reward_manager = getattr(self.unwrapped, "reward_manager", None)
        if reward_manager is None:
            raise RuntimeError("BonesRewardVectorWrapper requires an env with a reward_manager.")
        self._term_names = list(reward_manager.active_terms)
        _compute_reward_head_vector(
            reward_manager._step_reward,
            self._term_names,
            self.unwrapped.step_dt,
            self.reward_heads,
        )

    def reset(self, **kwargs):
        obs, extras = self.env.reset(**kwargs)
        extras = dict(extras)
        extras["reward_vector_names"] = list(self.reward_heads)
        return obs, extras

    def step(self, action):
        obs, reward, terminated, truncated, extras = self.env.step(action)
        reward_manager = self.unwrapped.reward_manager
        reward_vector, reward_names = _compute_reward_head_vector(
            reward_manager._step_reward,
            self._term_names,
            self.unwrapped.step_dt,
            self.reward_heads,
        )
        extras = dict(extras)
        extras["reward_vector"] = reward_vector
        extras["reward_vector_names"] = reward_names
        return obs, reward, terminated, truncated, extras


class MultiHeadPopArt(nn.Module):
    def __init__(
        self,
        num_heads: int,
        beta: float = 5.0e-4,
        debiased: bool = False,
        epsilon: float = 1.0e-5,
        min_sigma: float = 1.0e-4,
        max_sigma: float | None = None,
        stats_dtype: str = "float32",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.beta = beta
        self.debiased = debiased
        self.epsilon = epsilon
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        if stats_dtype not in {"float32", "float64"}:
            raise ValueError(f"Unsupported PopArt stats dtype: {stats_dtype}")
        self.stats_dtype = getattr(torch, stats_dtype)

        self.register_buffer("mu", torch.zeros(num_heads, dtype=self.stats_dtype))
        self.register_buffer("nu", torch.ones(num_heads, dtype=self.stats_dtype))
        self.register_buffer("sigma", torch.ones(num_heads, dtype=self.stats_dtype))
        if self.debiased:
            self.register_buffer("raw_mu", torch.zeros(num_heads, dtype=self.stats_dtype))
            self.register_buffer("raw_nu", torch.full((num_heads,), self.epsilon, dtype=self.stats_dtype))
            self.register_buffer("debias", torch.zeros(1, dtype=self.stats_dtype))

    def normalize(self, values: torch.Tensor) -> torch.Tensor:
        return ((values.to(self.mu.dtype) - self.mu) / self.sigma).to(values.dtype)

    def denormalize(self, values: torch.Tensor) -> torch.Tensor:
        return (values.to(self.mu.dtype) * self.sigma + self.mu).to(values.dtype)

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        targets = targets.to(self.mu.dtype)
        batch_mean = targets.mean(dim=0)
        batch_second_moment = targets.square().mean(dim=0)

        if self.debiased:
            self.raw_mu.mul_(1.0 - self.beta).add_(self.beta * batch_mean)
            self.raw_nu.mul_(1.0 - self.beta).add_(self.beta * batch_second_moment)
            self.debias.mul_(1.0 - self.beta).add_(self.beta)

            debias = self.debias.clamp(min=self.epsilon)
            self.mu.copy_(self.raw_mu / debias)
            self.nu.copy_(self.raw_nu / debias)
        else:
            self.mu.mul_(1.0 - self.beta).add_(self.beta * batch_mean)
            self.nu.mul_(1.0 - self.beta).add_(self.beta * batch_second_moment)

        variance = torch.clamp(self.nu - self.mu.square(), min=self.epsilon)
        self.sigma.copy_(variance.sqrt().clamp(min=self.min_sigma))
        if self.max_sigma is not None:
            self.sigma.clamp_(max=self.max_sigma)

        return old_mu, old_sigma

    @torch.no_grad()
    def preserve_output(self, layer: nn.Linear, old_mu: torch.Tensor, old_sigma: torch.Tensor):
        new_mu = self.mu.to(layer.weight.dtype)
        new_sigma = self.sigma.to(layer.weight.dtype)
        old_mu = old_mu.to(layer.weight.dtype)
        old_sigma = old_sigma.to(layer.weight.dtype)
        scale = (old_sigma / new_sigma).view(-1, 1)
        layer.weight.data.mul_(scale)
        layer.bias.data.copy_((old_sigma * layer.bias.data + old_mu - new_mu) / new_sigma)


class BonesPopArtActorCritic(nn.Module):
    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        value_head_names: list[str] | None = None,
        popart_beta: float = 5.0e-4,
        popart_debiased: bool = False,
        popart_epsilon: float = 1.0e-5,
        popart_min_sigma: float = 1.0e-4,
        popart_max_sigma: float | None = None,
        popart_stats_dtype: str = "float32",
        use_popart: bool = True,
        **kwargs,
    ):
        if kwargs:
            print(
                "BonesPopArtActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.value_head_names = value_head_names or list(DEFAULT_REWARD_HEADS)
        self.num_value_heads = len(self.value_head_names)
        self.use_popart = use_popart

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()

        self.critic = MLP(num_critic_obs, self.num_value_heads, critic_hidden_dims, activation)
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        if self.use_popart:
            self.popart = MultiHeadPopArt(
                self.num_value_heads,
                beta=popart_beta,
                debiased=popart_debiased,
                epsilon=popart_epsilon,
                min_sigma=popart_min_sigma,
                max_sigma=popart_max_sigma,
                stats_dtype=popart_stats_dtype,
            )
        else:
            self.popart = None

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        torch.distributions.Normal.set_default_validate_args(False)

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

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

    def update_distribution(self, obs):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        std = torch.clamp(std, min=0.3)
        self.distribution = torch.distributions.Normal(mean, std)

    def act(self, obs, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_obs)

    def evaluate_normalized(self, obs, **kwargs):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def evaluate(self, obs, denormalize: bool = True, **kwargs):
        values = self.evaluate_normalized(obs, **kwargs)
        if denormalize and self.popart is not None:
            return self.popart.denormalize(values)
        return values

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def get_critic_output_layer(self) -> nn.Linear:
        layer = self.critic[-1]
        if not isinstance(layer, nn.Linear):
            raise TypeError(f"Expected final critic layer to be nn.Linear, got: {type(layer)}")
        return layer

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class BonesRolloutStorage:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs,
        num_transitions_per_env,
        obs,
        actions_shape,
        num_value_heads,
        device="cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actions_shape = actions_shape
        self.num_value_heads = num_value_heads

        self.observations = RolloutStorage(
            "rl",
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            device,
        ).observations
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, num_value_heads, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, num_value_heads, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, num_value_heads, device=device)
        self.head_advantages = torch.zeros(num_transitions_per_env, num_envs, num_value_heads, device=device)
        self.actor_head_advantages = torch.zeros(num_transitions_per_env, num_envs, num_value_heads, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! You should call clear() before adding new transitions.")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_values, gamma, lam):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_is_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]
        self.head_advantages = self.returns - self.values
        self.actor_head_advantages = self.head_advantages.clone()

    def finalize_advantages(self, normalize_advantage: bool = True):
        flat_head_advantages = self.actor_head_advantages.flatten(0, 1)
        sigma, mu = torch.std_mean(flat_head_advantages, dim=0, unbiased=True)
        self.actor_head_advantages = (
            (self.actor_head_advantages - mu.view(1, 1, -1)) / (sigma.view(1, 1, -1) + 1e-8)
        )
        self.advantages = self.actor_head_advantages.sum(dim=-1, keepdim=True)
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        head_advantages = self.head_advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]
                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    head_advantages[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                )


class BonesPopArtPPO(PPO):
    def __init__(
        self,
        policy,
        value_head_names: list[str] | None = None,
        use_popart: bool = True,
        actor_advantage_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__(policy, **kwargs)
        self.transition = BonesRolloutStorage.Transition()
        self.reward_head_names = value_head_names or list(DEFAULT_REWARD_HEADS)
        self.use_popart = use_popart
        self.actor_advantage_reduction = actor_advantage_reduction
        if self.actor_advantage_reduction != "sum":
            raise ValueError(f"Only sum reduction is supported in v1, got: {self.actor_advantage_reduction}")
        self.latest_popart_stats = {}
        self.latest_advantage_stats = {}
        self.latest_value_stats = {}

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != "rl":
            raise ValueError("BonesPopArtPPO only supports RL training.")
        self.storage = BonesRolloutStorage(
            num_envs,
            num_transitions_per_env,
            obs,
            actions_shape,
            len(self.reward_head_names),
            self.device,
        )
        self.transition = BonesRolloutStorage.Transition()

    def act(self, obs):
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate(obs, denormalize=True).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)
        reward_vector = extras.get("reward_vector")
        if reward_vector is None:
            raise RuntimeError(
                "BonesPopArtPPO expected extras['reward_vector']. Wrap the env with BonesRewardVectorWrapper."
            )
        self.transition.rewards = reward_vector.to(self.device).clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device).unsqueeze(-1)
            self.transition.rewards += self.gamma * self.transition.values * time_outs

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_values = self.policy.evaluate(obs, denormalize=True).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

        if self.use_popart and self.policy.popart is not None:
            targets = self.storage.returns.view(-1, len(self.reward_head_names))
            old_mu, old_sigma = self.policy.popart.update_stats(targets)
            self.policy.popart.preserve_output(self.policy.get_critic_output_layer(), old_mu, old_sigma)
            self.storage.values = self.policy.popart.normalize(self.storage.values)
            self.storage.returns = self.policy.popart.normalize(self.storage.returns)
            self.storage.head_advantages = self.storage.returns - self.storage.values
            self.storage.actor_head_advantages = self.storage.head_advantages.clone()
            self.latest_popart_stats = {
                f"mu_{name}": self.policy.popart.mu[idx].item() for idx, name in enumerate(self.reward_head_names)
            }
            self.latest_popart_stats.update(
                {f"sigma_{name}": self.policy.popart.sigma[idx].item() for idx, name in enumerate(self.reward_head_names)}
            )
        else:
            self.latest_popart_stats = {}

        self.storage.finalize_advantages(normalize_advantage=not self.normalize_advantage_per_mini_batch)
        flat_actor_head_advantages = self.storage.actor_head_advantages.flatten(0, 1)
        head_abs_mean = flat_actor_head_advantages.abs().mean(dim=0)
        total_abs_mean = head_abs_mean.sum().clamp(min=1.0e-8)
        self.latest_advantage_stats = {}
        for idx, name in enumerate(self.reward_head_names):
            self.latest_advantage_stats[f"adv_mean_{name}"] = flat_actor_head_advantages[:, idx].mean().item()
            self.latest_advantage_stats[f"adv_std_{name}"] = flat_actor_head_advantages[:, idx].std(unbiased=True).item()
            self.latest_advantage_stats[f"adv_abs_mean_{name}"] = head_abs_mean[idx].item()
            self.latest_advantage_stats[f"adv_share_{name}"] = (head_abs_mean[idx] / total_abs_mean).item()

        flat_values = self.storage.values.flatten(0, 1)
        flat_returns = self.storage.returns.flatten(0, 1)
        value_residual = flat_returns - flat_values
        self.latest_value_stats = {
            "value_target_mean_abs": flat_returns.abs().mean().item(),
            "value_pred_mean_abs": flat_values.abs().mean().item(),
            "value_residual_mean_abs": value_residual.abs().mean().item(),
            "value_residual_max_abs": value_residual.abs().max().item(),
        }

    def update(self):  # noqa: C901
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        per_head_value_loss = torch.zeros(len(self.reward_head_names), device=self.device)

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            head_advantages_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            original_batch_size = obs_batch.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, denormalize=False, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean[:original_batch_size]
            sigma_batch = self.policy.action_std[:original_batch_size]
            entropy_batch = self.policy.entropy[:original_batch_size]

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
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss_terms = torch.max(value_losses, value_losses_clipped)
            else:
                value_loss_terms = (returns_batch - value_batch).pow(2)
            value_loss = value_loss_terms.mean()
            per_head_value_loss += value_loss_terms.mean(dim=0)

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

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        per_head_value_loss /= num_updates
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        for idx, name in enumerate(self.reward_head_names):
            loss_dict[f"value_function_{name}"] = per_head_value_loss[idx].item()
        for key, value in self.latest_popart_stats.items():
            loss_dict[f"popart_{key}"] = value
        for key, value in self.latest_advantage_stats.items():
            loss_dict[f"advantage_{key}"] = value
        for key, value in self.latest_value_stats.items():
            loss_dict[f"critic_{key}"] = value
        return loss_dict


class BonesOnPolicyRunner(OnPolicyRunner):
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str | None = None):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self.registry_name = registry_name

    def _construct_algorithm(self, obs):
        if not should_use_bones_popart_runner(self.cfg):
            return super()._construct_algorithm(obs)

        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.alg_cfg.get("rnd_cfg") is not None:
            raise NotImplementedError("Bones PopArt PPO v1 does not support RND.")
        if self.alg_cfg.get("symmetry_cfg") is not None:
            raise NotImplementedError("Bones PopArt PPO v1 does not support symmetry augmentation.")

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

        bones_cfg = _get_bones_popart_cfg(self.cfg)
        reward_heads = list(bones_cfg.get("reward_heads", DEFAULT_REWARD_HEADS))
        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name", None)
        policy_cfg["value_head_names"] = reward_heads
        policy_cfg["use_popart"] = bones_cfg.get("use_popart", True)
        policy_cfg["popart_beta"] = bones_cfg.get("beta", 5.0e-4)
        policy_cfg["popart_debiased"] = bones_cfg.get("debiased", False)
        policy_cfg["popart_epsilon"] = bones_cfg.get("epsilon", 1.0e-5)
        policy_cfg["popart_min_sigma"] = bones_cfg.get("min_sigma", 1.0e-4)
        policy_cfg["popart_max_sigma"] = bones_cfg.get("max_sigma")
        policy_cfg["popart_stats_dtype"] = bones_cfg.get("stats_dtype", "float32")

        actor_critic = BonesPopArtActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            **policy_cfg,
        ).to(self.device)

        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        alg_cfg.pop("rnd_cfg", None)
        alg_cfg.pop("symmetry_cfg", None)
        alg = BonesPopArtPPO(
            actor_critic,
            device=self.device,
            value_head_names=reward_heads,
            use_popart=bones_cfg.get("use_popart", True),
            actor_advantage_reduction=bones_cfg.get("actor_advantage_reduction", "sum"),
            multi_gpu_cfg=self.multi_gpu_cfg,
            **alg_cfg,
        )
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):  # noqa: C901
        if not should_use_bones_popart_runner(self.cfg):
            return super().learn(num_learning_iterations, init_at_random_ep_len)

        self._prepare_logging_writer()

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        reward_head_names = list(getattr(self.alg, "reward_head_names", DEFAULT_REWARD_HEADS))
        head_rewbuffers = {name: deque(maxlen=100) for name in reward_head_names}
        cur_head_reward_sum = torch.zeros(self.env.num_envs, len(reward_head_names), dtype=torch.float, device=self.device)
        rollout_head_reward_total = torch.zeros(len(reward_head_names), dtype=torch.float, device=self.device)
        rollout_head_reward_steps = 0

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations
        for it in range(start_iter, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    reward_vector = extras["reward_vector"].to(self.device)
                    rollout_head_reward_total += reward_vector.mean(dim=0)
                    rollout_head_reward_steps += 1

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])

                        cur_reward_sum += rewards
                        cur_head_reward_sum += reward_vector
                        cur_episode_length += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        done_ids = new_ids[:, 0]
                        rewbuffer.extend(cur_reward_sum[done_ids].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[done_ids].cpu().numpy().tolist())
                        for head_idx, head_name in enumerate(reward_head_names):
                            head_rewbuffers[head_name].extend(
                                cur_head_reward_sum[done_ids, head_idx].cpu().numpy().tolist()
                            )
                        cur_reward_sum[done_ids] = 0
                        cur_head_reward_sum[done_ids] = 0
                        cur_episode_length[done_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if rollout_head_reward_steps > 0:
                rollout_head_reward_mean = rollout_head_reward_total / rollout_head_reward_steps
            else:
                rollout_head_reward_mean = torch.zeros(len(reward_head_names), device=self.device)

            if self.log_dir is not None and not self.disable_logs:
                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            rollout_head_reward_total.zero_()
            rollout_head_reward_steps = 0

            if it == start_iter and not self.disable_logs:
                git_file_paths = []
                try:
                    from rsl_rl.utils import store_code_state

                    git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                except Exception:
                    git_file_paths = []
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width=width, pad=pad)
        if not should_use_bones_popart_runner(self.cfg):
            return

        reward_head_names = list(getattr(self.alg, "reward_head_names", DEFAULT_REWARD_HEADS))
        rollout_head_reward_mean = locs.get("rollout_head_reward_mean")
        if rollout_head_reward_mean is not None:
            for idx, head_name in enumerate(reward_head_names):
                self.writer.add_scalar(f"Train/reward_head_rollout/{head_name}", rollout_head_reward_mean[idx].item(), locs["it"])

        head_rewbuffers = locs.get("head_rewbuffers", {})
        for head_name, buffer in head_rewbuffers.items():
            if len(buffer) > 0:
                self.writer.add_scalar(f"Train/reward_head_episode/{head_name}", statistics.mean(buffer), locs["it"])

        popart = getattr(self.alg.policy, "popart", None)
        if popart is not None:
            for idx, head_name in enumerate(reward_head_names):
                self.writer.add_scalar(f"PopArt/mu/{head_name}", popart.mu[idx].item(), locs["it"])
                self.writer.add_scalar(f"PopArt/sigma/{head_name}", popart.sigma[idx].item(), locs["it"])

        loss_dict = locs.get("loss_dict", {})
        for idx, head_name in enumerate(reward_head_names):
            adv_mean = loss_dict.get(f"advantage_adv_mean_{head_name}")
            adv_std = loss_dict.get(f"advantage_adv_std_{head_name}")
            adv_share = loss_dict.get(f"advantage_adv_share_{head_name}")
            value_loss = loss_dict.get(f"value_function_{head_name}")
            if adv_mean is not None:
                self.writer.add_scalar(f"Advantage/mean/{head_name}", adv_mean, locs["it"])
            if adv_std is not None:
                self.writer.add_scalar(f"Advantage/std/{head_name}", adv_std, locs["it"])
            if adv_share is not None:
                self.writer.add_scalar(f"Advantage/share/{head_name}", adv_share, locs["it"])
            if value_loss is not None:
                self.writer.add_scalar(f"Loss/value_per_head/{head_name}", value_loss, locs["it"])

        critic_value_loss = loss_dict.get("value_function")
        if critic_value_loss is not None:
            self.writer.add_scalar("Loss/value", critic_value_loss, locs["it"])
        critic_residual = loss_dict.get("critic_value_residual_mean_abs")
        if critic_residual is not None:
            self.writer.add_scalar("Critic/value_residual_mean_abs", critic_residual, locs["it"])
        critic_residual_max = loss_dict.get("critic_value_residual_max_abs")
        if critic_residual_max is not None:
            self.writer.add_scalar("Critic/value_residual_max_abs", critic_residual_max, locs["it"])

    def save(self, path: str, infos=None):
        super().save(path, infos)
        if getattr(self, "logger_type", getattr(self, "_logger_type", None)) in ["wandb"]:
            import wandb

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

            if getattr(self, "registry_name", None) is not None and not self.registry_name.startswith("zarr:"):
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
