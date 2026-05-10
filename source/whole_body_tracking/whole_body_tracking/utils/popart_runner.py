"""PopArtMotionOnPolicyRunner — runner that wires up `ActorCriticPopArt` and `PPOPopArt`.

Subclasses `MotionOnPolicyRunner` and overrides `_construct_algorithm` to
bypass RSL-RL's bare-name `eval(class_name)` (which only resolves names in
the `rsl_rl.runners.on_policy_runner` module's scope) and instantiate our
PopArt-aware classes directly with their typed kwargs.

Selected by `agent_cfg.class_name = "PopArtMotionOnPolicyRunner"`; the
train script dispatches on that string.
"""

from __future__ import annotations

import torch

from whole_body_tracking.utils.actor_critic_popart import ActorCriticPopArt
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner
from whole_body_tracking.utils.ppo_popart import PPOPopArt


class PopArtMotionOnPolicyRunner(MotionOnPolicyRunner):
    def log(self, locs, width: int = 80, pad: int = 35):
        # Default RSL-RL logging (terminal printout + Loss/, Train/, Episode/, etc).
        super().log(locs, width=width, pad=pad)
        # PopArt diagnostics under their own wandb namespace.
        diag = getattr(self.alg, "popart_diagnostics", None)
        if diag and self.writer is not None:
            for key, value in diag.items():
                self.writer.add_scalar(f"popart/{key}", value, locs["it"])

    def _construct_algorithm(self, obs):
        # Resolve RND / symmetry hooks (we currently disallow both inside PPOPopArt,
        # but follow the parent's resolution path so a future activation is one-line).
        from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        # Honor the deprecated empirical_normalization shim if present.
        if self.cfg.get("empirical_normalization") is not None:
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        # Strip the class_name fields that RSL-RL would otherwise eval()
        # — we know the classes statically.
        self.policy_cfg.pop("class_name", None)
        self.alg_cfg.pop("class_name", None)

        # Pull PopArt-specific knobs off the policy cfg with sensible defaults.
        # num_categories is also popped (so it doesn't leak into ActorCritic kwargs)
        # but the env's command term is the source of truth — read it from there
        # whenever possible so cfg drift doesn't matter.
        cfg_num_categories = self.policy_cfg.pop("num_categories", None)
        popart_momentum: float = float(self.policy_cfg.pop("popart_momentum", 0.1))
        category_obs_group: str = str(self.policy_cfg.pop("category_obs_group", "category"))

        # Resolve num_categories and category_names from the command term.
        # Falls back to the agent cfg's num_categories if the term doesn't
        # expose it (which would only happen for non-popart command terms).
        num_categories = None
        category_names = None
        try:
            cmd_term = self.env.unwrapped.command_manager.get_term("motion")
            if hasattr(cmd_term, "num_categories"):
                num_categories = int(cmd_term.num_categories)
            if hasattr(cmd_term, "category_names"):
                category_names = list(cmd_term.category_names)
        except Exception:
            pass
        if num_categories is None:
            if cfg_num_categories is None:
                raise ValueError(
                    "PopArtMotionOnPolicyRunner requires num_categories on either "
                    "the command term (preferred) or agent_cfg.policy.num_categories."
                )
            num_categories = int(cfg_num_categories)

        actor_critic = ActorCriticPopArt(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            num_categories=num_categories,
            popart_momentum=popart_momentum,
            category_obs_group=category_obs_group,
            category_names=category_names,
            **self.policy_cfg,
        ).to(self.device)

        # Belt-and-suspenders: clipping in PopArt mode is unsupported (see ppo_popart.py).
        self.alg_cfg["use_clipped_value_loss"] = False

        alg = PPOPopArt(
            actor_critic,
            device=self.device,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

        alg.init_storage(
            "rl",
            self.env.num_envs,
            self.num_steps_per_env,
            obs,
            [self.env.num_actions],
        )
        return alg
