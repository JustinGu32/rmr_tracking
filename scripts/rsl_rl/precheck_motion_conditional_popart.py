"""Pre-check gate for motion-conditional PopArt on the full zarr dataset.

This script evaluates a trained scalar PPO multiclip Bones policy under
deterministic clip assignment and measures per-step post-weight reward statistics
for every (motion_id, reward_term) pair. It computes:

1. mean_spread_score:
   std_over_motions(mean_reward_per_motion) / mean_over_motions(std_reward_per_motion + eps)
2. sigma_spread_score:
   max_motion_std / max(min_motion_std, eps)

The gate passes if at least `min_terms_to_support` terms exceed the headline
threshold under either metric and all motions meet the minimum completed-episode
target.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Pre-check motion-conditional PopArt need on multiclip Bones policies.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the multiclip play task.")
parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr motion store.")
parser.add_argument("--max_clips", type=int, default=None, help="Max clips to load (None = all).")
parser.add_argument("--results_dir", type=str, default="eval_results", help="Directory to save output files.")
parser.add_argument("--results_name", type=str, default=None, help="Basename for output files.")
parser.add_argument(
    "--min_completed_episodes_per_motion",
    type=int,
    default=20,
    help="Minimum number of completed episodes to collect per motion.",
)
parser.add_argument(
    "--max_attempts_per_motion",
    type=int,
    default=50,
    help="Maximum number of episode attempts to collect per motion.",
)
parser.add_argument(
    "--min_terms_to_support",
    type=int,
    default=3,
    help="Minimum number of reward terms that must support the hypothesis to pass the gate.",
)
parser.add_argument(
    "--thresholds",
    type=str,
    default="2.0,3.0,5.0",
    help="Comma-separated thresholds for sensitivity analysis.",
)
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"], help="Policy activation function.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import zarr as _zarr

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


def load_zarr_metadata(zarr_path: str, exclude_objects: bool = True, max_clips: int | None = None):
    store = _zarr.open(zarr_path, mode="r")
    all_clip_start = store["clip_start_idx"][:]
    all_clip_end = store["clip_end_idx"][:]
    total_clips_raw = len(all_clip_start)

    all_names = [""] * total_clips_raw
    if "clip_names" in store:
        raw = store["clip_names"][:]
        for i in range(min(len(raw), total_clips_raw)):
            all_names[i] = str(raw[i])

    all_descs = [""] * total_clips_raw
    if "content_props_desc" in store:
        raw = store["content_props_desc"][:]
        for i in range(min(len(raw), total_clips_raw)):
            all_descs[i] = str(raw[i])

    exclude_props = ["object manipulation"] if exclude_objects else None
    if exclude_props and "content_props_desc" in store:
        valid_indices = []
        for i in range(total_clips_raw):
            d_str = all_descs[i].strip().lower()
            excluded = any(ep.lower() in d_str for ep in exclude_props)
            if not excluded:
                valid_indices.append(i)
    else:
        valid_indices = list(range(total_clips_raw))

    if max_clips is not None and max_clips < len(valid_indices):
        valid_indices = valid_indices[:max_clips]

    clip_names = [all_names[i] for i in valid_indices]
    clip_descs = [all_descs[i] for i in valid_indices]
    clip_lengths = [int(all_clip_end[i] - all_clip_start[i]) for i in valid_indices]
    num_clips = len(valid_indices)
    return clip_names, clip_descs, clip_lengths, num_clips


def parse_thresholds(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("At least one threshold must be provided.")
    return values


def compute_motion_term_stats(
    step_sum: torch.Tensor,
    step_sum_sq: torch.Tensor,
    step_count: torch.Tensor,
    episode_sum: torch.Tensor,
    episode_sum_sq: torch.Tensor,
    episode_count: torch.Tensor,
    thresholds: list[float],
    eps: float = 1.0e-8,
) -> tuple[list[dict], dict]:
    num_motions, num_terms = step_sum.shape
    per_term = []
    threshold_summary = {
        f"{threshold:.1f}": {
            "terms_passing_mean_spread": [],
            "terms_passing_sigma_spread": [],
            "terms_passing_either": [],
        }
        for threshold in thresholds
    }

    for term_idx in range(num_terms):
        counts = step_count[:, term_idx]
        valid_mask = counts > 0
        valid_count = int(valid_mask.sum().item())

        if valid_count > 0:
            means = torch.zeros(num_motions, dtype=torch.float64)
            stds = torch.zeros(num_motions, dtype=torch.float64)
            means[valid_mask] = step_sum[valid_mask, term_idx] / counts[valid_mask]
            vars_ = step_sum_sq[valid_mask, term_idx] / counts[valid_mask] - means[valid_mask].square()
            stds[valid_mask] = torch.sqrt(vars_.clamp_min(0.0))

            valid_means = means[valid_mask]
            valid_stds = stds[valid_mask]
            between_std = torch.std(valid_means, unbiased=valid_count > 1).item()
            mean_within_std = valid_stds.mean().item()
            min_motion_std = valid_stds.min().item()
            max_motion_std = valid_stds.max().item()
            mean_spread_score = between_std / (mean_within_std + eps)
            sigma_spread_score = max_motion_std / max(min_motion_std, eps)
        else:
            means = torch.zeros(num_motions, dtype=torch.float64)
            stds = torch.zeros(num_motions, dtype=torch.float64)
            between_std = 0.0
            mean_within_std = 0.0
            min_motion_std = 0.0
            max_motion_std = 0.0
            mean_spread_score = 0.0
            sigma_spread_score = 0.0

        ep_counts = episode_count[:, term_idx]
        ep_valid_mask = ep_counts > 0
        ep_means = torch.zeros(num_motions, dtype=torch.float64)
        ep_stds = torch.zeros(num_motions, dtype=torch.float64)
        if int(ep_valid_mask.sum().item()) > 0:
            ep_means[ep_valid_mask] = episode_sum[ep_valid_mask, term_idx] / ep_counts[ep_valid_mask]
            ep_vars = episode_sum_sq[ep_valid_mask, term_idx] / ep_counts[ep_valid_mask] - ep_means[ep_valid_mask].square()
            ep_stds[ep_valid_mask] = torch.sqrt(ep_vars.clamp_min(0.0))

        threshold_flags = {}
        for threshold in thresholds:
            passes_mean = mean_spread_score >= threshold
            passes_sigma = sigma_spread_score >= threshold
            passes_either = passes_mean or passes_sigma
            threshold_flags[f"{threshold:.1f}"] = {
                "passes_mean_spread": passes_mean,
                "passes_sigma_spread": passes_sigma,
                "passes_either": passes_either,
            }

        term_report = {
            "term_index": term_idx,
            "valid_motion_count": valid_count,
            "between_motion_mean_std": between_std,
            "mean_within_motion_std": mean_within_std,
            "min_motion_std": min_motion_std,
            "max_motion_std": max_motion_std,
            "mean_spread_score": mean_spread_score,
            "sigma_spread_score": sigma_spread_score,
            "threshold_flags": threshold_flags,
            "per_motion_step_mean": means.tolist(),
            "per_motion_step_std": stds.tolist(),
            "per_motion_episode_sum_mean": ep_means.tolist(),
            "per_motion_episode_sum_std": ep_stds.tolist(),
            "per_motion_step_count": counts.tolist(),
            "per_motion_episode_count": ep_counts.tolist(),
        }
        per_term.append(term_report)

    return per_term, threshold_summary


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.policy.activation = args_cli.activation
    env_cfg.scene.num_envs = args_cli.num_envs
    thresholds = parse_thresholds(args_cli.thresholds)

    print("[INFO] This pre-check expects a trained scalar PPO checkpoint, not a PopArt checkpoint.")

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        wandb_run_id = wandb_run.id
        wandb_run_name = wandb_run.name

        files = [f.name for f in wandb_run.files() if "model" in f.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{file}"
        checkpoint_source = f"{run_path}/{file}"
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        wandb_run_id = "local"
        wandb_run_name = "local"
        checkpoint_source = resume_path

    zarr_path = args_cli.zarr_path
    if zarr_path is None:
        zarr_path = getattr(env_cfg.commands.motion, "zarr_path", "")
    assert zarr_path and os.path.isdir(zarr_path), f"--zarr_path required and must exist: {zarr_path}"
    env_cfg.commands.motion.zarr_path = zarr_path

    if args_cli.max_clips is not None and hasattr(env_cfg.commands.motion, "max_clips"):
        env_cfg.commands.motion.max_clips = args_cli.max_clips

    exclude_objects = getattr(env_cfg.commands.motion, "exclude_objects", True)
    clip_names, clip_descs, clip_lengths, num_clips = load_zarr_metadata(
        zarr_path, exclude_objects=exclude_objects, max_clips=args_cli.max_clips
    )
    print(f"[INFO] Loaded {num_clips} clips from zarr store for pre-check analysis.")

    env_cfg.episode_length_s = 120.0
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "force_push_robot"):
            env_cfg.events.force_push_robot = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    reward_manager = raw_env.reward_manager
    term_manager = raw_env.termination_manager
    motion_cmd = raw_env.command_manager.get_term("motion")
    device = raw_env.device

    if not hasattr(motion_cmd, "clip_ids") or not hasattr(motion_cmd.motion, "num_clips"):
        raise RuntimeError("This pre-check requires the zarr multiclip motion command path.")

    reward_term_names = list(reward_manager.active_terms)
    reward_term_weights = [float(reward_manager.get_term_cfg(name).weight) for name in reward_term_names]
    num_terms = len(reward_term_names)

    assert num_clips == motion_cmd.motion.num_clips, (
        f"Metadata clip count ({num_clips}) != loader clip count ({motion_cmd.motion.num_clips})."
    )

    results_dir = args_cli.results_dir
    os.makedirs(results_dir, exist_ok=True)
    if args_cli.results_name:
        base_name = args_cli.results_name
    else:
        base_name = f"popart_motion_precheck_{wandb_run_id}"
    episodes_path = os.path.join(results_dir, f"{base_name}_episodes.jsonl")
    summary_path = os.path.join(results_dir, f"{base_name}_summary.json")

    header = {
        "type": "header",
        "task": args_cli.task,
        "checkpoint_source": checkpoint_source,
        "wandb_path": args_cli.wandb_path,
        "wandb_run_name": wandb_run_name if args_cli.wandb_path else None,
        "zarr_path": zarr_path,
        "num_clips_total": num_clips,
        "num_envs": args_cli.num_envs,
        "reward_term_names": reward_term_names,
        "reward_term_weights": reward_term_weights,
        "min_completed_episodes_per_motion": args_cli.min_completed_episodes_per_motion,
        "max_attempts_per_motion": args_cli.max_attempts_per_motion,
        "thresholds": thresholds,
    }
    with open(episodes_path, "w") as f:
        f.write(json.dumps(header) + "\n")

    completed_events: list[dict] = []
    episode_ids = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    env_clip_id = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    env_episode_steps = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    env_episode_term_sums = torch.zeros(args_cli.num_envs, num_terms, dtype=torch.float64, device=device)
    episode_id_counter = 0
    assignment_cursor = 0

    motion_attempt_counts = torch.zeros(num_clips, dtype=torch.long, device=device)
    motion_completed_counts = torch.zeros(num_clips, dtype=torch.long, device=device)
    motion_success_counts = torch.zeros(num_clips, dtype=torch.long, device=device)
    motion_failure_counts = torch.zeros(num_clips, dtype=torch.long, device=device)
    motion_step_counts = torch.zeros(num_clips, num_terms, dtype=torch.long, device=device)
    motion_step_sum = torch.zeros(num_clips, num_terms, dtype=torch.float64, device=device)
    motion_step_sum_sq = torch.zeros(num_clips, num_terms, dtype=torch.float64, device=device)
    motion_episode_sum = torch.zeros(num_clips, num_terms, dtype=torch.float64, device=device)
    motion_episode_sum_sq = torch.zeros(num_clips, num_terms, dtype=torch.float64, device=device)
    motion_episode_count = torch.zeros(num_clips, num_terms, dtype=torch.long, device=device)

    def next_eligible_clip() -> int:
        nonlocal assignment_cursor
        for offset in range(num_clips):
            cid = (assignment_cursor + offset) % num_clips
            if (
                motion_completed_counts[cid].item() < args_cli.min_completed_episodes_per_motion
                and motion_attempt_counts[cid].item() < args_cli.max_attempts_per_motion
            ):
                assignment_cursor = (cid + 1) % num_clips
                return cid
        return -1

    def assign_clip_to_env(eid: int) -> bool:
        nonlocal episode_id_counter
        cid = next_eligible_clip()
        if cid < 0:
            env_clip_id[eid] = -1
            episode_ids[eid] = -1
            return False

        motion_attempt_counts[cid] += 1
        env_clip_id[eid] = cid
        episode_ids[eid] = episode_id_counter
        episode_id_counter += 1
        motion_cmd.clip_ids[eid] = cid
        motion_cmd.clip_start[eid] = motion_cmd.motion.clip_start_idx[cid]
        motion_cmd.clip_end[eid] = motion_cmd.motion.clip_end_idx[cid]
        motion_cmd.time_steps[eid] = motion_cmd.motion.clip_start_idx[cid]
        return True

    def _write_robot_state(cmd, env_ids_t):
        from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, sample_uniform

        cmd._cache_current_frames()
        root_pos = cmd.body_pos_w[:, 0].clone()
        root_ori = cmd.body_quat_w[:, 0].clone()
        root_lin_vel = cmd.body_lin_vel_w[:, 0].clone()
        root_ang_vel = cmd.body_ang_vel_w[:, 0].clone()
        range_list = [cmd.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=cmd.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_t), 6), device=cmd.device)
        root_pos[env_ids_t] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids_t] = quat_mul(orientations_delta, root_ori[env_ids_t])
        range_list = [cmd.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=cmd.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_t), 6), device=cmd.device)
        root_lin_vel[env_ids_t] += rand_samples[:, :3]
        root_ang_vel[env_ids_t] += rand_samples[:, 3:]
        joint_pos = cmd.joint_pos.clone()
        joint_vel = cmd.joint_vel.clone()
        joint_pos += sample_uniform(*cmd.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = cmd.robot.data.soft_joint_pos_limits[env_ids_t]
        joint_pos[env_ids_t] = torch.clip(
            joint_pos[env_ids_t], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        cmd.robot.write_joint_state_to_sim(joint_pos[env_ids_t], joint_vel[env_ids_t], env_ids=env_ids_t)
        cmd.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids_t], root_ori[env_ids_t], root_lin_vel[env_ids_t], root_ang_vel[env_ids_t]], dim=-1),
            env_ids=env_ids_t,
        )

    import types
    from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

    def _eval_resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        active_to_write = []
        for eid in env_ids_t.tolist():
            old_cid = int(env_clip_id[eid].item())
            if old_cid < 0:
                continue
            per_term = term_manager._term_dones[eid]
            fired = [term_manager._term_names[i] for i in range(len(term_manager._term_names)) if per_term[i]]
            term_reason = fired[0] if fired else "unknown"
            completed_events.append(
                {
                    "eid": eid,
                    "clip_id": old_cid,
                    "episode_id": int(episode_ids[eid].item()),
                    "termination_reason": term_reason,
                    "success": False,
                }
            )
            if assign_clip_to_env(eid):
                active_to_write.append(eid)
        if active_to_write:
            _write_robot_state(self, torch.tensor(active_to_write, device=self.device, dtype=torch.long))

    def _eval_update_command(self):
        self.time_steps += 1
        inactive = env_clip_id < 0
        if inactive.any():
            self.time_steps[inactive] = 0

        clip_end_ids = torch.where(self.time_steps >= self.clip_end)[0]
        if len(clip_end_ids) > 0:
            active_to_write = []
            for eid in clip_end_ids.tolist():
                old_cid = int(env_clip_id[eid].item())
                if old_cid < 0:
                    continue
                completed_events.append(
                    {
                        "eid": eid,
                        "clip_id": old_cid,
                        "episode_id": int(episode_ids[eid].item()),
                        "termination_reason": "clip_end",
                        "success": True,
                    }
                )
                if assign_clip_to_env(eid):
                    active_to_write.append(eid)
            if active_to_write:
                _write_robot_state(self, torch.tensor(active_to_write, device=self.device, dtype=torch.long))

        self._cache_current_frames()
        n_bodies = len(self.cfg.body_names)
        anchor_pos_w_exp = self.anchor_pos_w[:, None, :].expand(-1, n_bodies, -1)
        anchor_quat_w_exp = self.anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)
        robot_anchor_quat_w_exp = self.robot_anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)
        delta_pos_w = self.robot_anchor_pos_w[:, None, :].expand(-1, n_bodies, -1).clone()
        delta_pos_w[..., 2] = anchor_pos_w_exp[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_exp, quat_inv(anchor_quat_w_exp)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_exp)

    motion_cmd._resample_command = types.MethodType(_eval_resample_command, motion_cmd)
    motion_cmd._update_command = types.MethodType(_eval_update_command, motion_cmd)

    for eid in range(args_cli.num_envs):
        assign_clip_to_env(eid)

    def _init_resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        active_ids = env_ids_t[env_clip_id[env_ids_t] >= 0]
        if len(active_ids) > 0:
            _write_robot_state(self, active_ids)

    motion_cmd._resample_command = types.MethodType(_init_resample_command, motion_cmd)
    obs, _ = env.reset()
    motion_cmd._resample_command = types.MethodType(_eval_resample_command, motion_cmd)

    motion_cmd._cache_current_frames()
    n_bodies = len(motion_cmd.cfg.body_names)
    anchor_pos_w_exp = motion_cmd.anchor_pos_w[:, None, :].expand(-1, n_bodies, -1)
    anchor_quat_w_exp = motion_cmd.anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)
    robot_anchor_quat_w_exp = motion_cmd.robot_anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)
    delta_pos_w = motion_cmd.robot_anchor_pos_w[:, None, :].expand(-1, n_bodies, -1).clone()
    delta_pos_w[..., 2] = anchor_pos_w_exp[..., 2]
    delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_exp, quat_inv(anchor_quat_w_exp)))
    motion_cmd.body_quat_relative_w = quat_mul(delta_ori_w, motion_cmd.body_quat_w)
    motion_cmd.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, motion_cmd.body_pos_w - anchor_pos_w_exp)

    print("[INFO] Starting motion-conditional PopArt pre-check evaluation.")

    def all_done() -> bool:
        needs_more = (
            (motion_completed_counts < args_cli.min_completed_episodes_per_motion)
            & (motion_attempt_counts < args_cli.max_attempts_per_motion)
        )
        return not bool(needs_more.any().item()) and bool((env_clip_id < 0).all().item())

    while simulation_app.is_running() and not all_done():
        prev_clip_id = env_clip_id.clone()
        prev_episode_ids = episode_ids.clone()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        step_rewards = reward_manager._step_reward.detach().to(dtype=torch.float64).clone()
        active_prev = prev_clip_id >= 0
        if active_prev.any():
            env_episode_term_sums[active_prev] += step_rewards[active_prev]
            env_episode_steps[active_prev] += 1

            for cid in prev_clip_id[active_prev].unique().tolist():
                cid_mask = prev_clip_id == cid
                cid_rewards = step_rewards[cid_mask]
                motion_step_counts[cid] += cid_mask.sum()
                motion_step_sum[cid] += cid_rewards.sum(dim=0)
                motion_step_sum_sq[cid] += cid_rewards.square().sum(dim=0)

        if completed_events:
            pending_events = completed_events.copy()
            completed_events.clear()
            for event in pending_events:
                eid = event["eid"]
                cid = event["clip_id"]
                if cid < 0:
                    continue
                motion_completed_counts[cid] += 1
                if event["success"]:
                    motion_success_counts[cid] += 1
                else:
                    motion_failure_counts[cid] += 1

                motion_episode_sum[cid] += env_episode_term_sums[eid]
                motion_episode_sum_sq[cid] += env_episode_term_sums[eid].square()
                motion_episode_count[cid] += 1

                episode_record = {
                    "type": "episode",
                    "motion_id": cid,
                    "clip_id": cid,
                    "clip_name": clip_names[cid],
                    "clip_desc": clip_descs[cid],
                    "clip_length_frames": clip_lengths[cid],
                    "episode_id": event["episode_id"],
                    "steps": int(env_episode_steps[eid].item()),
                    "attempt_index_for_motion": int(motion_attempt_counts[cid].item()),
                    "completed_count_for_motion": int(motion_completed_counts[cid].item()),
                    "success": event["success"],
                    "termination_reason": event["termination_reason"],
                    "term_episode_sums_post_weight": {
                        reward_term_names[idx]: env_episode_term_sums[eid, idx].item() for idx in range(num_terms)
                    },
                }
                with open(episodes_path, "a") as f:
                    f.write(json.dumps(episode_record) + "\n")

                env_episode_term_sums[eid].zero_()
                env_episode_steps[eid] = 0

        if int(motion_completed_counts.sum().item()) > 0 and int(motion_completed_counts.sum().item()) % 50 < args_cli.num_envs:
            completed_total = int(motion_completed_counts.sum().item())
            target_total = num_clips * args_cli.min_completed_episodes_per_motion
            print(f"[INFO] Completed episodes: {completed_total}/{target_total}")

    per_term_stats, threshold_summary = compute_motion_term_stats(
        step_sum=motion_step_sum,
        step_sum_sq=motion_step_sum_sq,
        step_count=motion_step_counts,
        episode_sum=motion_episode_sum,
        episode_sum_sq=motion_episode_sum_sq,
        episode_count=motion_episode_count,
        thresholds=thresholds,
    )

    term_reports = []
    for idx, term_stats in enumerate(per_term_stats):
        term_stats["term_name"] = reward_term_names[idx]
        term_stats["reward_weight"] = reward_term_weights[idx]
        term_reports.append(term_stats)
        for threshold in thresholds:
            threshold_key = f"{threshold:.1f}"
            flags = term_stats["threshold_flags"][threshold_key]
            if flags["passes_mean_spread"]:
                threshold_summary[threshold_key]["terms_passing_mean_spread"].append(reward_term_names[idx])
            if flags["passes_sigma_spread"]:
                threshold_summary[threshold_key]["terms_passing_sigma_spread"].append(reward_term_names[idx])
            if flags["passes_either"]:
                threshold_summary[threshold_key]["terms_passing_either"].append(reward_term_names[idx])

    motion_reports = []
    for cid in range(num_clips):
        motion_reports.append(
            {
                "motion_id": cid,
                "clip_id": cid,
                "clip_name": clip_names[cid],
                "clip_desc": clip_descs[cid],
                "clip_length_frames": clip_lengths[cid],
                "attempt_count": int(motion_attempt_counts[cid].item()),
                "completed_episode_count": int(motion_completed_counts[cid].item()),
                "success_count": int(motion_success_counts[cid].item()),
                "failure_count": int(motion_failure_counts[cid].item()),
                "met_min_completed_episode_target": bool(
                    motion_completed_counts[cid].item() >= args_cli.min_completed_episodes_per_motion
                ),
                "exhausted_attempt_budget": bool(
                    motion_attempt_counts[cid].item() >= args_cli.max_attempts_per_motion
                ),
            }
        )

    term_reports_sorted_by_mean = sorted(term_reports, key=lambda item: item["mean_spread_score"], reverse=True)
    term_reports_sorted_by_sigma = sorted(term_reports, key=lambda item: item["sigma_spread_score"], reverse=True)
    headline_threshold_key = "3.0" if 3.0 in thresholds else f"{thresholds[0]:.1f}"
    sample_count_sufficient = all(report["met_min_completed_episode_target"] for report in motion_reports)
    num_terms_supporting_mean = len(threshold_summary[headline_threshold_key]["terms_passing_mean_spread"])
    num_terms_supporting_sigma = len(threshold_summary[headline_threshold_key]["terms_passing_sigma_spread"])
    num_terms_supporting_either = len(threshold_summary[headline_threshold_key]["terms_passing_either"])
    proceed = sample_count_sufficient and (num_terms_supporting_either >= args_cli.min_terms_to_support)

    summary = {
        "type": "summary",
        "task": args_cli.task,
        "checkpoint_source": checkpoint_source,
        "wandb_path": args_cli.wandb_path,
        "wandb_run_name": wandb_run_name if args_cli.wandb_path else None,
        "zarr_path": zarr_path,
        "num_clips_total": num_clips,
        "num_envs": args_cli.num_envs,
        "reward_term_names": reward_term_names,
        "reward_term_weights": reward_term_weights,
        "thresholds": thresholds,
        "headline_threshold": float(headline_threshold_key),
        "min_completed_episodes_per_motion": args_cli.min_completed_episodes_per_motion,
        "max_attempts_per_motion": args_cli.max_attempts_per_motion,
        "min_terms_to_support": args_cli.min_terms_to_support,
        "sample_count_sufficient": sample_count_sufficient,
        "proceed_with_motion_conditional_popart": proceed,
        "num_terms_supporting_mean_spread": num_terms_supporting_mean,
        "num_terms_supporting_sigma_spread": num_terms_supporting_sigma,
        "num_terms_supporting_either": num_terms_supporting_either,
        "threshold_summary": threshold_summary,
        "per_term_ranked_by_mean_spread": term_reports_sorted_by_mean,
        "per_term_ranked_by_sigma_spread": term_reports_sorted_by_sigma,
        "per_motion_sample_counts": motion_reports,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("MOTION-CONDITIONAL POPART PRE-CHECK")
    print("=" * 80)
    print(f"Sample count sufficient: {sample_count_sufficient}")
    print(f"Proceed with motion-conditional PopArt: {proceed}")
    print(f"Terms supporting mean spread @ {headline_threshold_key}: {num_terms_supporting_mean}")
    print(f"Terms supporting sigma spread @ {headline_threshold_key}: {num_terms_supporting_sigma}")
    print(f"Terms supporting either @ {headline_threshold_key}: {num_terms_supporting_either}")
    print("\nTop terms by mean_spread_score:")
    for item in term_reports_sorted_by_mean[:10]:
        print(f"  {item['term_name']:35s} mean={item['mean_spread_score']:.3f} sigma={item['sigma_spread_score']:.3f}")
    print("\nTop terms by sigma_spread_score:")
    for item in term_reports_sorted_by_sigma[:10]:
        print(f"  {item['term_name']:35s} sigma={item['sigma_spread_score']:.3f} mean={item['mean_spread_score']:.3f}")
    print(f"\n[INFO] Episode records saved to: {episodes_path}")
    print(f"[INFO] Summary saved to: {summary_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
