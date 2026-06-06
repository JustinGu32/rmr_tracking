import math
import os
import time
import warnings

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.modules import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
from whole_body_tracking.utils.depth_encoder_actor_critic import DepthCNNActorCritic, DepthEncoderActorCritic
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


# ─── Activation Health Tracker ────────────────────────────────────────────────

class ActivationHealthTracker:
    """Track activation health metrics for MLP hidden layers.

    Logs per-layer:
      - effective_rank: SVD-based expressivity (higher = more diverse representations)
      - elu_saturation: fraction of neurons stuck in ELU negative saturation

    Registers forward hooks on Linear layers. Gated by `self.enabled`.
    """

    def __init__(self, model: nn.Module, sat_threshold: float = -0.9):
        self.sat_threshold = sat_threshold
        self.enabled = False
        self._hooks = []
        self._activations = {}  # layer_name -> list of activation tensors

        # Hook on hidden Linear layers (skip output layer)
        linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
        for name, module in linear_layers[:-1]:
            key = f"hidden_{name}"
            hook = module.register_forward_hook(self._make_hook(key))
            self._hooks.append(hook)
            self._activations[key] = []

    def _make_hook(self, layer_name: str):
        def hook_fn(module, inp, out):
            if not self.enabled:
                return
            self._activations[layer_name].append(out.detach().cpu())
        return hook_fn

    def reset(self):
        for key in self._activations:
            self._activations[key] = []

    def compute(self) -> dict[str, float]:
        results = {}
        for i, (name, act_list) in enumerate(self._activations.items()):
            if not act_list:
                continue
            activations = torch.cat(act_list, dim=0)  # (total_samples, neurons)

            # Effective rank (SVD-based)
            try:
                _, S, _ = torch.linalg.svd(activations, full_matrices=False)
                S = S / S.sum()
                entropy = -(S * (S + 1e-8).log()).sum()
                results[f"activation_health/effective_rank_layer_{i}"] = entropy.exp().item()
            except Exception:
                pass

            # ELU saturation fraction
            mean_acts = activations.mean(dim=0)
            saturated = (mean_acts < self.sat_threshold).float().mean()
            results[f"activation_health/elu_saturation_layer_{i}"] = saturated.item()

        return results

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# # ─── Activation Health Tracker (v5 — dormant neuron recycling) ────────────────
#
# class ActivationHealthTracker:
#     """Track activation health and recycle dormant neurons.
#
#     Logs per-layer:
#       - effective_rank: SVD-based expressivity
#       - dormant_fraction: fraction of neurons with low mean |activation|
#
#     Optionally recycles dormant neurons by re-initializing their weights.
#     Registers forward hooks on Linear layers. Gated by `self.enabled`.
#     """
#
#     def __init__(self, model: nn.Module, dormant_threshold: float = 0.025):
#         self.dormant_threshold = dormant_threshold
#         self.enabled = False
#         self._hooks = []
#         self._activations = {}  # layer_name -> list of activation tensors
#         self._linear_layers = []  # (name, module) pairs for recycling
#
#         # Hook on hidden Linear layers (skip output layer)
#         linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
#         for name, module in linear_layers[:-1]:
#             key = f"hidden_{name}"
#             hook = module.register_forward_hook(self._make_hook(key))
#             self._hooks.append(hook)
#             self._activations[key] = []
#             self._linear_layers.append((key, module))
#
#         # Keep reference to next layer for zeroing outgoing weights
#         self._next_linear = {}
#         for i in range(len(linear_layers) - 1):
#             key = f"hidden_{linear_layers[i][0]}"
#             self._next_linear[key] = linear_layers[i + 1][1]
#
#     def _make_hook(self, layer_name: str):
#         def hook_fn(module, inp, out):
#             if not self.enabled:
#                 return
#             self._activations[layer_name].append(out.detach().cpu())
#         return hook_fn
#
#     def reset(self):
#         for key in self._activations:
#             self._activations[key] = []
#
#     def compute(self) -> dict[str, float]:
#         results = {}
#         for i, (name, act_list) in enumerate(self._activations.items()):
#             if not act_list:
#                 continue
#             activations = torch.cat(act_list, dim=0)  # (total_samples, neurons)
#
#             # Effective rank (SVD-based)
#             try:
#                 _, S, _ = torch.linalg.svd(activations, full_matrices=False)
#                 S = S / S.sum()
#                 entropy = -(S * (S + 1e-8).log()).sum()
#                 results[f"activation_health/effective_rank_layer_{i}"] = entropy.exp().item()
#             except Exception:
#                 pass
#
#             # Dormant fraction: neurons with very low mean |activation|
#             mean_abs_act = activations.abs().mean(dim=0)
#             dormant = (mean_abs_act < self.dormant_threshold).float().mean()
#             results[f"activation_health/dormant_fraction_layer_{i}"] = dormant.item()
#
#         return results
#
#     @torch.no_grad()
#     def recycle(self) -> dict[str, float]:
#         """Re-initialize dormant neurons. Call after compute().
#
#         For each dormant neuron:
#           - Re-init its incoming weights (orthogonal) and zero bias
#           - Zero its outgoing weights in the next layer so the reset
#             doesn't disrupt the network immediately
#
#         Returns dict with per-layer recycled counts.
#         """
#         results = {}
#         for i, (name, act_list) in enumerate(self._activations.items()):
#             if not act_list:
#                 continue
#             activations = torch.cat(act_list, dim=0)
#             mean_abs_act = activations.abs().mean(dim=0)
#             dormant_mask = mean_abs_act < self.dormant_threshold
#             n_dormant = dormant_mask.sum().item()
#             results[f"activation_health/recycled_layer_{i}"] = n_dormant
#
#             if n_dormant == 0:
#                 continue
#
#             # Find the Linear layer and its successor
#             _, linear = self._linear_layers[i]
#             next_linear = self._next_linear.get(name)
#
#             dormant_idx = dormant_mask.nonzero(as_tuple=True)[0]
#             device = linear.weight.device
#
#             # Re-init incoming weights for dormant neurons (orthogonal)
#             fan_in = linear.weight.shape[1]
#             for idx in dormant_idx:
#                 new_weight = torch.empty(1, fan_in, device=device)
#                 nn.init.orthogonal_(new_weight)
#                 linear.weight.data[idx] = new_weight[0]
#                 linear.bias.data[idx] = 0.0
#
#             # Zero outgoing weights so recycled neurons start quiet
#             if next_linear is not None:
#                 next_linear.weight.data[:, dormant_idx] = 0.0
#
#         return results
#
#     def remove_hooks(self):
#         for h in self._hooks:
#             h.remove()
#         self._hooks.clear()


# ─── Runners ──────────────────────────────────────────────────────────────────

class MyOnPolicyRunner(OnPolicyRunner):
    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if self.logger_type in ["wandb"]:
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            # v5: use self.alg.get_policy() instead of self.alg.policy
            policy = self.alg.get_policy() if hasattr(self.alg, 'get_policy') else self.alg.policy
            normalizer = policy.obs_normalizers.get("actor") if hasattr(policy, 'obs_normalizers') else getattr(policy, 'actor_obs_normalizer', None)
            export_policy_as_onnx(policy, normalizer=normalizer, path=policy_path, filename=filename)
            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class MotionOnPolicyRunner(OnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        # Strip deprecated fields that IsaacLab config serializes but rsl_rl v5 doesn't accept
        _deprecated_model_keys = ["stochastic", "init_noise_std", "noise_std_type", "state_dependent_std",
                                  "obs_normalization", "actor_obs_normalization", "critic_obs_normalization"]
        for section in ["actor", "critic"]:
            if section in train_cfg:
                for key in _deprecated_model_keys:
                    train_cfg[section].pop(key, None)
        # The standard PPO runner should not receive Bones PopArt-only algorithm flags.
        for key in [
            "use_popart_multihead",
            "popart_hierarchical",
            "popart_head_mode",
            "popart_groups",
            "popart_group_preset",
            "popart_grouped_actor_weight_mode",
            "popart_momentum",
            "popart_epsilon",
            "popart_normalize_actor_weights",
            "popart_actor_advantage_scaling",
        ]:
            train_cfg.get("algorithm", {}).pop(key, None)
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

        # Set up activation health tracking on the actor MLP (if available)
        policy = self.alg.get_policy() if hasattr(self.alg, 'get_policy') else self.alg.policy
        if hasattr(policy, 'mlp'):
            self.dormant_tracker = ActivationHealthTracker(policy.mlp)
        else:
            self.dormant_tracker = None
        self._dormant_measure_interval = 100  # measure every N iterations

    # # learn() override for newer rsl_rl with self.logger API — commented out for current version
    # def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    #     """Run the learning loop with dormant neuron tracking."""
    #     if init_at_random_ep_len:
    #         self.env.episode_length_buf = torch.randint_like(
    #             self.env.episode_length_buf, high=int(self.env.max_episode_length)
    #         )
    #
    #     obs = self.env.get_observations().to(self.device)
    #     self.alg.train_mode()
    #
    #     if self.is_distributed:
    #         print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
    #         self.alg.broadcast_parameters()
    #
    #     self.logger.init_logging_writer()
    #
    #     start_it = self.current_learning_iteration
    #     total_it = start_it + num_learning_iterations
    #     for it in range(start_it, total_it):
    #         start = time.time()
    #
    #         # Enable dormant neuron tracking only on measurement iterations
    #         is_measure_iter = (it % self._dormant_measure_interval == 0) and self.dormant_tracker is not None
    #         if is_measure_iter:
    #             self.dormant_tracker.reset()
    #             self.dormant_tracker.enabled = True
    #
    #         # Rollout
    #         with torch.inference_mode():
    #             for _ in range(self.cfg["num_steps_per_env"]):
    #                 actions = self.alg.act(obs)
    #                 obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
    #                 if self.cfg.get("check_for_nan", True):
    #                     from rsl_rl.utils import check_nan
    #                     check_nan(obs, rewards, dones)
    #                 obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
    #                 self.alg.process_env_step(obs, rewards, dones, extras)
    #                 intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
    #                 self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
    #
    #             stop = time.time()
    #             collect_time = stop - start
    #             start = stop
    #             self.alg.compute_returns(obs)
    #
    #         # Update policy
    #         loss_dict = self.alg.update()
    #
    #         stop = time.time()
    #         learn_time = stop - start
    #         self.current_learning_iteration = it
    #
    #         # Log information
    #         self.logger.log(
    #             it=it, start_it=start_it, total_it=total_it,
    #             collect_time=collect_time, learn_time=learn_time,
    #             loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
    #             action_std=self.alg.get_policy().output_std,
    #             rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
    #         )
    #
    #         # Log dormant neuron stats and disable tracking
    #         if is_measure_iter and wandb.run is not None:
    #             dormant_stats = self.dormant_tracker.compute()
    #             self.dormant_tracker.enabled = False
    #             wandb.log(dormant_stats, step=it)
    #
    #         # Save model
    #         if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
    #             self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
    #
    #     if self.logger.writer is not None:
    #         self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
    #         self.logger.stop_logging_writer()

    def _get_depth_start_idx(self) -> int:
        """Return the flat index at which 'depth_image' starts in the policy obs vector."""
        obs_manager = self.env.unwrapped.observation_manager
        term_names = obs_manager.active_terms.get("policy", [])
        term_dims = obs_manager.group_obs_term_dim.get("policy", [])
        start = 0
        for name, shape in zip(term_names, term_dims):
            if name == "depth_image":
                return start
            start += math.prod(shape)
        raise RuntimeError(
            "Could not find 'depth_image' term in the policy observation group. "
            "Pass --depth_obs together with --depth_encoder."
        )

    def _construct_algorithm(self, obs):
        """Construct the PPO algorithm, using a depth-encoder ActorCritic when enabled.

        --depth_encoder  → DepthEncoderActorCritic (MLP encoder, WBT_USE_DEPTH_ENCODER)
        --depth_cnn      → DepthCNNActorCritic     (CNN encoder, WBT_USE_DEPTH_CNN)
        Both require --depth_obs (WBT_USE_DEPTH_OBS=1).
        """
        depth_obs_on = os.environ.get("WBT_USE_DEPTH_OBS", "0") == "1"
        use_depth_encoder = depth_obs_on and os.environ.get("WBT_USE_DEPTH_ENCODER", "0") == "1"
        use_depth_cnn = depth_obs_on and os.environ.get("WBT_USE_DEPTH_CNN", "0") == "1"

        if use_depth_encoder and use_depth_cnn:
            raise ValueError("WBT_USE_DEPTH_ENCODER and WBT_USE_DEPTH_CNN are mutually exclusive.")

        if not (use_depth_encoder or use_depth_cnn):
            return super()._construct_algorithm(obs)

        actor_critic_cls = DepthCNNActorCritic if use_depth_cnn else DepthEncoderActorCritic

        # ── depth-encoder path ─────────────────────────────────────────────
        self.alg_cfg = resolve_rnd_config(self.alg_cfg, obs, self.cfg["obs_groups"], self.env)
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Use "
                "`actor_obs_normalization` / `critic_obs_normalization` in the policy config.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("actor_obs_normalization") is None:
                self.policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if self.policy_cfg.get("critic_obs_normalization") is None:
                self.policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        depth_start_idx = self._get_depth_start_idx()
        print(f"[DepthEncoderActorCritic] depth_start_idx resolved from obs manager: {depth_start_idx}")

        policy_cfg = dict(self.policy_cfg)
        policy_cfg.pop("class_name", None)

        actor_critic = actor_critic_cls(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            depth_start_idx=depth_start_idx,
            **policy_cfg,
        ).to(self.device)

        alg_cfg = dict(self.alg_cfg)
        alg_cfg.pop("class_name", None)
        alg = PPO(actor_critic, device=self.device, **alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg)
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def save(self, path: str, infos=None):
        """Save the model and training information."""
        super().save(path, infos)
        if getattr(self, "logger_type", getattr(self, "_logger_type", None)) in ["wandb"]:
            cmd = self.env.unwrapped.command_manager.get_term("motion")
            is_multiclip = hasattr(cmd.cfg, "zarr_path") and cmd.cfg.zarr_path
            policy_path = path.split("model")[0]
            filename = policy_path.split("/")[-2] + ".onnx"
            policy = self.alg.policy
            normalizer = getattr(policy, 'actor_obs_normalizer', None)
            if is_multiclip:
                # Policy-only ONNX (no motion data) to avoid 2GB protobuf limit
                export_policy_as_onnx(policy, normalizer=normalizer, path=policy_path, filename=filename)
            else:
                export_motion_policy_as_onnx(
                    self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
                )

            # # v5 save() — uses get_policy() and obs_normalizers
            # policy_path = path.split("model")[0]
            # filename = policy_path.split("/")[-2] + ".onnx"
            # policy = self.alg.get_policy() if hasattr(self.alg, 'get_policy') else self.alg.policy
            # normalizer = policy.obs_normalizers.get("actor") if hasattr(policy, 'obs_normalizers') else getattr(policy, 'actor_obs_normalizer', None)
            # export_motion_policy_as_onnx(
            #     self.env.unwrapped, policy, normalizer=normalizer, path=policy_path, filename=filename
            # )

            attach_onnx_metadata(self.env.unwrapped, wandb.run.name, path=policy_path, filename=filename)
            wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))

            # link the artifact registry to this run (skip for zarr paths)
            if self.registry_name is not None and not self.registry_name.startswith("zarr:"):
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
