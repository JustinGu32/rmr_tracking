import os
import time

import numpy as np
import torch
import torch.nn as nn

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

# ── Add swish/silu activation support to rsl_rl (its resolver lacks it) ──────────
# rsl_rl.utils.resolve_nn_activation only knows elu/relu/etc. We wrap it to map
# "swish"/"silu" -> nn.SiLU, and reassign the name on every module that already
# did `from rsl_rl.utils import resolve_nn_activation` at import time.
import rsl_rl.utils as _rsl_utils
import rsl_rl.modules as _rsl_modules

_orig_resolve = _rsl_utils.resolve_nn_activation


def _resolve_with_swish(act_name: str):
    if act_name in ("swish", "silu"):
        return nn.SiLU()
    return _orig_resolve(act_name)


_rsl_utils.resolve_nn_activation = _resolve_with_swish
for _modname in ("actor_critic", "actor_critic_recurrent", "student_teacher",
                 "student_teacher_recurrent", "rnd"):
    _mod = getattr(_rsl_modules, _modname, None)
    if _mod is not None and hasattr(_mod, "resolve_nn_activation"):
        _mod.resolve_nn_activation = _resolve_with_swish

from isaaclab_rl.rsl_rl import export_policy_as_onnx

import wandb
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

        # ── Video logging to WandB ──────────────────────────────────────────────
        # Enabled via env vars (set by train.py's --video flag). Renders a single
        # deterministic full-motion rollout of env 0 every N iters and uploads it
        # as wandb.Video. Requires the app launched with rendering (ENABLE_CAMERAS=1).
        self._video_enabled = os.environ.get("WBT_VIDEO", "0") == "1"
        self._video_interval = int(os.environ.get("WBT_VIDEO_INTERVAL", "1000"))
        # Max frames to render (safety cap); the rollout normally stops at end-of-motion.
        self._video_max_frames = int(os.environ.get("WBT_VIDEO_LENGTH", "600"))
        self._video_warmup = 4  # throwaway renders so the RTX buffer isn't black
        # playback fps = control rate = 1 / (decimation * physics_dt)
        try:
            self._video_fps = int(round(1.0 / self.env.unwrapped.step_dt))
        except Exception:
            self._video_fps = 30

    def _log_video(self, it: int):
        """Render one deterministic full-motion rollout of env 0 and log it to WandB.

        Steps ALL envs (the sim is a single batched env) but only records env 0's
        follow-camera frames. env 0 is forced to start at motion frame 0 and the
        rollout runs until the motion clip ends (my_time_out) or the safety cap.
        """
        env = self.env.unwrapped
        if env.render_mode != "rgb_array":
            return  # app was not launched with rendering; nothing to capture

        cmd = env.command_manager.get_term("motion")
        T = int(cmd.motion.time_step_total)
        policy = self.alg.act_inference if hasattr(self.alg, "act_inference") else self.alg.policy.act_inference

        frames = []
        # rsl_rl's rollout runs under inference_mode, which makes env tensors "inference
        # tensors" that cannot be modified outside inference_mode. Match that context here.
        with torch.inference_mode():
            obs, _ = self.env.get_observations()
            # Force env 0 to the start of the clip so we capture a full climb.
            cmd.time_steps[0] = 0

            # Warm the renderer (first few frames are black until the RTX buffer fills).
            for _ in range(self._video_warmup):
                env.render()

            for _ in range(min(self._video_max_frames, T + 2)):
                actions = policy(obs.to(self.device))
                obs, _, _, _ = self.env.step(actions.to(self.env.device))
                frame = env.render()
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))
                # Stop once env 0 has played through the whole clip.
                if int(cmd.time_steps[0]) >= T - 1:
                    break

        if not frames:
            return
        # (T, H, W, 3) -> (T, 3, H, W) for wandb.Video
        video = np.stack(frames).transpose(0, 3, 1, 2)
        try:
            wandb.log({"video/rollout": wandb.Video(video, fps=self._video_fps, format="mp4")}, step=it)
        except Exception as e:
            print(f"[WARN] video upload failed: {e}")

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)
        it = locs["it"]
        # Skip it=0: don't spend the expensive render rollout on a fresh random policy before
        # the renderer is even warm (and it would block the first training print for minutes).
        if (
            self._video_enabled
            and wandb.run is not None
            and self._video_interval > 0
            and it > 0
            and it % self._video_interval == 0
        ):
            self._log_video(it)

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

            # link the artifact registry to this run (skip for zarr / local paths)
            if (
                self.registry_name is not None
                and not self.registry_name.startswith("zarr:")
                and not self.registry_name.startswith("local:")
            ):
                wandb.run.use_artifact(self.registry_name)
                self.registry_name = None
