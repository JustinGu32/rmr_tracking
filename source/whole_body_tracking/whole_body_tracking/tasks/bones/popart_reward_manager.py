from __future__ import annotations

import torch

from isaaclab.managers.reward_manager import RewardManager


PER_TERM_REWARDS_RAW_KEY = "per_term_rewards_raw"
PER_HEAD_REWARDS_KEY = "per_head_rewards"
WEIGHTED_STEP_REWARDS_KEY = "weighted_step_rewards"
REWARD_WEIGHTS_KEY = "reward_weights"
REWARD_TERM_NAMES_KEY = "reward_term_names"


class BonesPerTermRewardManager(RewardManager):
    """Reward manager that exposes unweighted per-term rewards for PopArt training."""

    def __init__(self, cfg: object, env):
        super().__init__(cfg, env)
        self.reward_weights = torch.tensor(
            [float(term_cfg.weight) for term_cfg in self._term_cfgs], dtype=torch.float, device=self.device
        )
        self._step_reward_raw = torch.zeros((self.num_envs, len(self._term_names)), dtype=torch.float, device=self.device)
        self._publish_static_extras()

    def _publish_static_extras(self):
        self._env.extras[REWARD_WEIGHTS_KEY] = self.reward_weights
        self._env.extras[REWARD_TERM_NAMES_KEY] = list(self._term_names)
        self._env.extras[WEIGHTED_STEP_REWARDS_KEY] = self._step_reward
        if PER_TERM_REWARDS_RAW_KEY not in self._env.extras:
            self._env.extras[PER_TERM_REWARDS_RAW_KEY] = self._step_reward_raw
        if PER_HEAD_REWARDS_KEY not in self._env.extras:
            self._env.extras[PER_HEAD_REWARDS_KEY] = self._step_reward_raw

    def compute(self, dt: float) -> torch.Tensor:
        self._reward_buf[:] = 0.0

        for term_idx, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            raw_value = term_cfg.func(self._env, **term_cfg.params) * dt
            weighted_value = raw_value * float(term_cfg.weight)

            self._reward_buf += weighted_value
            self._episode_sums[name] += weighted_value
            self._step_reward[:, term_idx] = weighted_value / dt
            self._step_reward_raw[:, term_idx] = raw_value

        self._publish_static_extras()
        self._env.extras[PER_TERM_REWARDS_RAW_KEY] = self._step_reward_raw
        self._env.extras[PER_HEAD_REWARDS_KEY] = self._step_reward_raw
        self._env.extras[WEIGHTED_STEP_REWARDS_KEY] = self._step_reward
        return self._reward_buf


def install_bones_per_term_reward_manager(env) -> BonesPerTermRewardManager:
    """Replace the environment reward manager with the PopArt-aware bones variant."""

    unwrapped = env.unwrapped
    reward_manager = getattr(unwrapped, "reward_manager", None)
    if reward_manager is None:
        raise AttributeError("Bones PopArt requires an environment with a reward_manager.")
    if isinstance(reward_manager, BonesPerTermRewardManager):
        reward_manager._publish_static_extras()
        return reward_manager

    popart_reward_manager = BonesPerTermRewardManager(unwrapped.cfg.rewards, unwrapped)
    unwrapped.reward_manager = popart_reward_manager
    return popart_reward_manager
