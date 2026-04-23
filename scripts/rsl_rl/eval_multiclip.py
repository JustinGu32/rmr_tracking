"""Per-clip evaluation of a multiclip policy.

Runs a trained policy on every clip in a Zarr store (one at a time, starting
from t=0) and records per-clip diagnostics:
  - clip name / description (motion type)
  - which termination fired (or success if it reached the end)
  - how far through the clip the robot got (% and absolute frames)
  - per-clip tracking error means (anchor pos, body pos, etc.)
  - adaptive sampling bin-failure histogram snapshot

Results are saved to a JSON file for offline analysis.

Usage:
    python scripts/rsl_rl/eval_multiclip.py \
        --task=Tracking-MultiClip-Flat-G1-Play-v0 \
        --zarr_path=/path/to/motions.zarr \
        --wandb_path=org/project/run_id \
        --num_envs=50 --headless \
        --results_name=my_eval
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Per-clip multiclip policy evaluation.")
parser.add_argument("--num_envs", type=int, default=50, help="Number of parallel environments.")
parser.add_argument("--task", type=str, default=None, help="Name of the task (use a MultiClip Play variant).")
parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr motion store.")
parser.add_argument("--max_clips", type=int, default=None, help="Max clips to load (None = all).")
parser.add_argument("--results_dir", type=str, default="eval_results", help="Directory to save result JSON.")
parser.add_argument("--results_name", type=str, default=None, help="Filename (without .json).")
parser.add_argument("--disable_terminations", action="store_true", default=False,
                    help="Disable failure terminations (only stop at clip end). Useful to see full tracking error curves.")
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"],
                    help="Activation function for actor/critic networks (default: elu).")
parser.add_argument("--include_motion_types", type=str, default=None,
                    help="Comma-separated keywords to restrict eval clips by clip_name (case-insensitive substring, OR semantics). "
                         "Mirror the value used at train time to eval on the same motions.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import json
import torch
import zarr as _zarr

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401


def load_zarr_metadata(zarr_path: str, exclude_objects: bool = True, max_clips: int | None = None,
                       include_keywords: list[str] | None = None):
    """Load clip names, descriptions, and frame counts from Zarr store.

    Applies the SAME filtering as ZarrMotionLoader so that internal clip IDs
    (0, 1, 2, ...) map to the correct metadata.
    """
    store = _zarr.open(zarr_path, mode="r")
    all_clip_start = store["clip_start_idx"][:]
    all_clip_end = store["clip_end_idx"][:]
    total_clips_raw = len(all_clip_start)

    # Read all metadata arrays
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

    # Apply the same exclude_props filter as ZarrMotionLoader
    exclude_props = [
        "object manipulation", "wall", "chair", "obstacle",
        "edge", "safety pad", "railing", "box",
    ] if exclude_objects else None
    if exclude_props and "content_props_desc" in store:
        valid_indices = []
        for i in range(total_clips_raw):
            d_str = all_descs[i].strip().lower()
            excluded = any(ep.lower() in d_str for ep in exclude_props)
            if not excluded:
                valid_indices.append(i)
        excluded_count = total_clips_raw - len(valid_indices)
        print(f"[load_zarr_metadata] Excluded {excluded_count}/{total_clips_raw} clips "
              f"matching props: {exclude_props}")
    else:
        valid_indices = list(range(total_clips_raw))

    # Apply the same include_keywords filter as ZarrMotionLoader (after exclude, before max)
    if include_keywords:
        kws = [kw.lower() for kw in include_keywords]
        before = len(valid_indices)
        valid_indices = [i for i in valid_indices
                         if any(kw in all_names[i].lower() for kw in kws)]
        print(f"[load_zarr_metadata] Kept {len(valid_indices)}/{before} clips "
              f"matching keywords: {include_keywords}")

    # Apply max_clips limit (same as ZarrMotionLoader)
    if max_clips is not None and max_clips < len(valid_indices):
        valid_indices = valid_indices[:max_clips]

    # Build filtered metadata using valid_indices
    clip_names = [all_names[i] for i in valid_indices]
    clip_descs = [all_descs[i] for i in valid_indices]
    clip_lengths = [int(all_clip_end[i] - all_clip_start[i]) for i in valid_indices]
    num_clips = len(valid_indices)

    return clip_names, clip_descs, clip_lengths, num_clips


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Run policy on every clip and collect per-clip diagnostics."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.policy.activation = args_cli.activation
    env_cfg.scene.num_envs = args_cli.num_envs

    # --- Load model checkpoint ---
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
        print(f"[INFO] Loaded checkpoint: {run_path}/{file}")
    else:
        log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        wandb_run_id = "local"
        wandb_run_name = "local"
        print(f"[INFO] Loaded checkpoint: {resume_path}")

    # --- Set zarr path ---
    zarr_path = args_cli.zarr_path
    if zarr_path is None:
        zarr_path = getattr(env_cfg.commands.motion, "zarr_path", "")
    assert zarr_path and os.path.isdir(zarr_path), f"--zarr_path required and must exist: {zarr_path}"
    env_cfg.commands.motion.zarr_path = zarr_path

    if args_cli.max_clips is not None and hasattr(env_cfg.commands.motion, "max_clips"):
        env_cfg.commands.motion.max_clips = args_cli.max_clips

    # Plumb keyword filter into env_cfg so the loader applies it identically
    include_keywords = None
    if args_cli.include_motion_types is not None:
        include_keywords = [s.strip() for s in args_cli.include_motion_types.split(",") if s.strip()]
        if hasattr(env_cfg.commands.motion, "include_motion_types"):
            env_cfg.commands.motion.include_motion_types = include_keywords
            print(f"[EVAL] Restricting clips by motion-type keywords: {include_keywords}")

    # --- Load metadata (with same filtering as ZarrMotionLoader) ---
    exclude_objects = getattr(env_cfg.commands.motion, "exclude_objects", True)
    clip_names, clip_descs, clip_lengths, num_clips = load_zarr_metadata(
        zarr_path, exclude_objects=exclude_objects, max_clips=args_cli.max_clips,
        include_keywords=include_keywords,
    )
    print(f"\n[EVAL] {num_clips} clips in zarr store, {sum(clip_lengths)} total frames")

    # --- Force play-mode settings: start from t=0, long episodes, no perturbations ---
    env_cfg.episode_length_s = 120.0  # Long enough for any clip
    # Disable perturbation events if they exist
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "force_push_robot"):
            env_cfg.events.force_push_robot = None

    # Always disable xy-position failure termination during eval
    if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "bad_anchor_pos_xy"):
        env_cfg.terminations.bad_anchor_pos_xy = None
        print("[EVAL] Disabled bad_anchor_pos_xy termination")

    # Optionally disable failure terminations (keep only time_out)
    if args_cli.disable_terminations and hasattr(env_cfg, "terminations"):
        term_cfg = env_cfg.terminations
        for attr in dir(term_cfg):
            if attr.startswith("_") or attr == "time_out":
                continue
            if hasattr(getattr(term_cfg, attr, None), "func"):
                setattr(term_cfg, attr, None)
        print("[EVAL] Disabled all failure terminations (only time_out active)")

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # --- Load policy ---
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    term_manager = raw_env.termination_manager
    term_names = term_manager._term_names
    motion_cmd = raw_env.command_manager.get_term("motion")
    metric_names = list(motion_cmd.metrics.keys())
    device = env.unwrapped.device

    # Sanity check: our metadata clip count must match what ZarrMotionLoader actually loaded
    assert num_clips == motion_cmd.motion.num_clips, (
        f"Metadata clip count ({num_clips}) != loader clip count ({motion_cmd.motion.num_clips}). "
        f"Filtering mismatch — check exclude_objects setting."
    )

    print(f"[EVAL] Termination terms: {term_names}")
    print(f"[EVAL] Tracking metrics: {metric_names}")
    print(f"[EVAL] Num envs: {args_cli.num_envs}\n")

    # --- Compute results path early (so we can save incrementally) ---
    results_dir = args_cli.results_dir
    os.makedirs(results_dir, exist_ok=True)
    if args_cli.results_name:
        filename = f"{args_cli.results_name}.jsonl"
    else:
        tag = wandb_run_id if args_cli.wandb_path else "local"
        filename = f"multiclip_eval_{tag}.jsonl"
    results_path = os.path.join(results_dir, filename)

    # Write header line with metadata
    header = {
        "type": "header",
        "task": args_cli.task,
        "wandb_path": args_cli.wandb_path,
        "wandb_run_name": wandb_run_name if args_cli.wandb_path else None,
        "zarr_path": zarr_path,
        "num_clips_total": num_clips,
        "num_envs": args_cli.num_envs,
    }
    with open(results_path, "w") as f:
        f.write(json.dumps(header) + "\n")

    # --- Per-clip results storage ---
    clip_results = []
    clips_evaluated = 0
    clips_remaining = list(range(num_clips))
    # Envs reassigned via clip-end in _eval_update_command during the current step.
    # The main loop must skip metric accumulation for these because motion_cmd.metrics
    # still contains the OLD clip's final metrics (from _update_metrics which ran
    # before _update_command detected clip_end).
    _clip_end_reassigned = set()

    # Per-env tracking: which clip is assigned, accumulated metrics, step count
    env_clip_id = torch.full((args_cli.num_envs,), -1, dtype=torch.long, device=device)
    env_step_count = torch.zeros(args_cli.num_envs, dtype=torch.long, device=device)
    env_metric_sums = {name: torch.zeros(args_cli.num_envs, device=device) for name in metric_names}
    env_max_errors = {
        "max_anchor_pos_err": torch.zeros(args_cli.num_envs, device=device),
        "max_body_pos_err": torch.zeros(args_cli.num_envs, device=device),
    }

    def assign_clip_to_env(eid: int):
        """Assign next unprocessed clip to a single env. Returns True if a clip was assigned."""
        if not clips_remaining:
            env_clip_id[eid] = -1
            return False
        cid = clips_remaining.pop(0)
        env_clip_id[eid] = cid
        motion_cmd.clip_ids[eid] = cid
        motion_cmd.clip_start[eid] = motion_cmd.motion.clip_start_idx[cid]
        motion_cmd.clip_end[eid] = motion_cmd.motion.clip_end_idx[cid]
        motion_cmd.time_steps[eid] = motion_cmd.motion.clip_start_idx[cid]
        # Reset per-env accumulators
        env_step_count[eid] = 0
        for name in metric_names:
            env_metric_sums[name][eid] = 0.0
        for k in env_max_errors:
            env_max_errors[k][eid] = 0.0
        return True

    def record_clip_result(eid: int, term_reason: str):
        """Save results for a completed clip evaluation."""
        nonlocal clips_evaluated
        cid = int(env_clip_id[eid].item())
        if cid < 0:
            return
        steps = int(env_step_count[eid].item())
        total_len = clip_lengths[cid]
        progress = steps / max(total_len, 1)
        result = {
            "clip_id": cid,
            "clip_name": clip_names[cid],
            "clip_desc": clip_descs[cid],
            "clip_length_frames": total_len,
            "steps_survived": steps,
            "progress_pct": round(progress * 100, 1),
            "termination_reason": term_reason,
            "success": term_reason == "clip_end",
            # "metrics": {},
            # "max_errors": {},
        }
        # for name in metric_names:
        #     mean_val = env_metric_sums[name][eid].item() / max(steps, 1)
        #     result["metrics"][name] = round(mean_val, 6)
        # for k, v in env_max_errors.items():
        #     result["max_errors"][k] = round(v[eid].item(), 6)
        clip_results.append(result)
        clips_evaluated += 1
        # Append this result as a single line to the JSONL file
        with open(results_path, "a") as f:
            f.write(json.dumps(result) + "\n")

    def _write_robot_state(cmd, env_ids_t):
        """Write robot physical state to match reference at current time_steps."""
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

    # --- Monkey-patch _resample_command and _update_command ---
    #
    # There are TWO paths that end a clip inside env.step():
    #   1. Framework reset (failure termination) → command_manager.reset() → _resample_command
    #   2. _update_command detects time_steps >= clip_end → calls _resample_command
    #      (this does NOT trigger a framework termination — the env just keeps going)
    #
    # Both paths originally call _adaptive_sampling to pick a new random clip/timestep.
    # We replace them so they record the result and assign the next eval clip instead.
    #
    # Order within env.step():
    #   a) Physics sim
    #   b) Compute terminations
    #   c) _reset_idx for terminated envs → _resample_command (path 1)
    #   d) command_manager.compute() → _update_command → _resample_command (path 2)
    #   e) Compute observations

    import types
    from isaaclab.utils.math import quat_from_euler_xyz, quat_inv, quat_mul, quat_apply, sample_uniform, yaw_quat

    def _flush_metrics_for_env(eid):
        """Accumulate the current step's metrics before recording the result.

        Called only from _eval_update_command (clip-end path), where
        _update_metrics() has already run for the current step inside
        CommandTerm.compute(), so motion_cmd.metrics contains fresh data.

        NOT called from _eval_resample_command (failure path), because that
        runs during _reset_idx before compute() — metrics are stale from the
        previous step and were already accumulated by the main loop.
        """
        env_step_count[eid] += 1
        for name in metric_names:
            env_metric_sums[name][eid] += motion_cmd.metrics[name][eid]
        if "error_anchor_pos" in motion_cmd.metrics:
            env_max_errors["max_anchor_pos_err"][eid] = max(
                env_max_errors["max_anchor_pos_err"][eid].item(),
                motion_cmd.metrics["error_anchor_pos"][eid].item(),
            )
        if "error_body_pos" in motion_cmd.metrics:
            env_max_errors["max_body_pos_err"][eid] = max(
                env_max_errors["max_body_pos_err"][eid].item(),
                motion_cmd.metrics["error_body_pos"][eid].item(),
            )

    def _eval_resample_command(self, env_ids):
        """Called by framework on termination reset. Record failure, assign next clip.

        Runs during _reset_idx, BEFORE command_manager.compute(). Metrics are
        stale (previous step), already accumulated by the main loop — don't flush.
        """
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        for eid in env_ids_t.tolist():
            if env_clip_id[eid] < 0:
                continue
            per_term = term_manager._term_dones[eid]
            fired = [term_names[i] for i in range(len(term_names)) if per_term[i]]
            term_reason = fired[0] if fired else "unknown"
            record_clip_result(eid, term_reason)
            assign_clip_to_env(eid)

        # Write robot state for newly assigned clips
        active_ids = env_ids_t[env_clip_id[env_ids_t] >= 0]
        if len(active_ids) > 0:
            _write_robot_state(self, active_ids)

    def _eval_update_command(self):
        """Patched _update_command that records clip-end as success."""
        self.time_steps += 1

        # Clamp time_steps for inactive envs (clip_id == -1) so they never
        # index out of bounds in _cache_current_frames().  Without this,
        # inactive envs' time_steps grows unbounded and eventually exceeds
        # the motion array length, causing a CUDA index-out-of-bounds assert.
        inactive = env_clip_id < 0
        if inactive.any():
            self.time_steps[inactive] = 0

        # Detect envs that reached clip end (success — played the full clip)
        clip_end_ids = torch.where(self.time_steps >= self.clip_end)[0]
        if len(clip_end_ids) > 0:
            for eid in clip_end_ids.tolist():
                if env_clip_id[eid] < 0:
                    continue
                _flush_metrics_for_env(eid)
                record_clip_result(eid, "clip_end")
                assign_clip_to_env(eid)
                # Mark so main loop skips accumulation — motion_cmd.metrics
                # still holds the OLD clip's values from _update_metrics()
                _clip_end_reassigned.add(eid)

            # Write robot state for newly assigned clips
            active_ids = clip_end_ids[env_clip_id[clip_end_ids] >= 0]
            if len(active_ids) > 0:
                _write_robot_state(self, active_ids)

        # Refresh the per-step frame cache
        self._cache_current_frames()

        # Compute relative body poses (same as original)
        n_bodies = len(self.cfg.body_names)
        anchor_pos_w_exp = self.anchor_pos_w[:, None, :].expand(-1, n_bodies, -1)
        anchor_quat_w_exp = self.anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)
        robot_anchor_quat_w_exp = self.robot_anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)
        delta_pos_w = self.robot_anchor_pos_w[:, None, :].expand(-1, n_bodies, -1).clone()
        delta_pos_w[..., 2] = anchor_pos_w_exp[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_exp, quat_inv(anchor_quat_w_exp)))
        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_exp)

        # Skip adaptive sampling EMA update (not needed for eval)

    motion_cmd._resample_command = types.MethodType(_eval_resample_command, motion_cmd)
    motion_cmd._update_command = types.MethodType(_eval_update_command, motion_cmd)
    print("[EVAL] Monkey-patched _resample_command and _update_command for per-clip eval")

    # --- Initial assignment ---
    for eid in range(args_cli.num_envs):
        assign_clip_to_env(eid)

    # Reset env — the patched _resample_command will be called for all envs,
    # but env_clip_id is already set so it will just write robot state correctly.
    # However, reset() calls command_manager.reset() which calls _resample_command
    # BEFORE our assign_clip_to_env has run... so we need to assign first, then reset.
    # We already assigned above. But reset() will call _resample_command which will
    # call record_clip_result on the just-assigned clips (wrong!).
    #
    # To handle this: temporarily use a simpler resample that just writes state.
    def _init_resample_command(self, env_ids):
        """During initial reset, just write robot state without recording."""
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        active_ids = env_ids_t[env_clip_id[env_ids_t] >= 0]
        if len(active_ids) > 0:
            _write_robot_state(self, active_ids)

    motion_cmd._resample_command = types.MethodType(_init_resample_command, motion_cmd)
    obs, _ = env.reset()
    # Restore the real eval resample after initial reset
    motion_cmd._resample_command = types.MethodType(_eval_resample_command, motion_cmd)

    # Warm up body_pos_relative_w: env.reset() only calls command_manager.reset()
    # (which writes robot state), but NOT command_manager.compute() (which computes
    # body_pos_relative_w). Without this, body_pos_relative_w stays as zeros from
    # __init__, causing the first termination check to see huge Z-errors on wrists/
    # ankles and immediately terminate all standing/walking clips.
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

    print(f"[EVAL] Starting evaluation of {num_clips} clips...")

    # --- Main eval loop ---
    # Recording and clip assignment happen inside the monkey-patched methods
    # during env.step(). The main loop just runs the policy and tracks metrics.
    while clips_evaluated < num_clips and simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # Track steps and accumulate metrics for active envs, EXCLUDING envs
        # that were just reassigned via clip-end (their metrics are stale from
        # the old clip — _update_metrics ran before _update_command detected clip_end).
        active = env_clip_id >= 0
        if _clip_end_reassigned:
            skip_mask = torch.zeros(args_cli.num_envs, dtype=torch.bool, device=device)
            for eid in _clip_end_reassigned:
                skip_mask[eid] = True
            accumulate = active & ~skip_mask
            _clip_end_reassigned.clear()
        else:
            accumulate = active

        env_step_count[accumulate] += 1
        for name in metric_names:
            env_metric_sums[name][accumulate] += motion_cmd.metrics[name][accumulate]
        if "error_anchor_pos" in motion_cmd.metrics:
            env_max_errors["max_anchor_pos_err"][accumulate] = torch.max(
                env_max_errors["max_anchor_pos_err"][accumulate],
                motion_cmd.metrics["error_anchor_pos"][accumulate]
            )
        if "error_body_pos" in motion_cmd.metrics:
            env_max_errors["max_body_pos_err"][accumulate] = torch.max(
                env_max_errors["max_body_pos_err"][accumulate],
                motion_cmd.metrics["error_body_pos"][accumulate]
            )

        # Print progress when clip count crosses a multiple of 10
        if clips_evaluated > 0 and clips_evaluated % 10 < args_cli.num_envs and clips_evaluated != getattr(main, '_last_print', -1):
            main._last_print = clips_evaluated
            successes_so_far = sum(1 for r in clip_results if r["success"])
            print(f"  [{clips_evaluated}/{num_clips}] clips done, {successes_so_far} successes, "
                  f"avg progress: {sum(r['progress_pct'] for r in clip_results) / len(clip_results):.1f}%")

    # --- Dump adaptive sampling state ---
    # sampling_state = {}
    # if hasattr(motion_cmd, "bin_failed_count"):
    #     bfc = motion_cmd.bin_failed_count.cpu().tolist()
    #     sampling_state["bin_failed_count"] = bfc
    #     sampling_state["bin_count"] = motion_cmd.bin_count
    #     sampling_state["total_frames"] = motion_cmd.total_frames

    # --- Aggregate statistics ---
    successes = [r for r in clip_results if r["success"]]
    failures = [r for r in clip_results if not r["success"]]

    # Group failures by termination reason
    failure_by_reason = {}
    for r in failures:
        reason = r["termination_reason"]
        if reason not in failure_by_reason:
            failure_by_reason[reason] = []
        failure_by_reason[reason].append(r)

    # Group by motion type (clip description)
    by_desc = {}
    for r in clip_results:
        desc = r["clip_desc"] or "unknown"
        if desc not in by_desc:
            by_desc[desc] = {"count": 0, "successes": 0, "total_progress": 0.0}
        by_desc[desc]["count"] += 1
        by_desc[desc]["successes"] += int(r["success"])
        by_desc[desc]["total_progress"] += r["progress_pct"]

    # Find worst clips (lowest progress)
    sorted_by_progress = sorted(clip_results, key=lambda r: r["progress_pct"])

    # --- Print summary ---
    print("\n" + "=" * 70)
    print(f"MULTICLIP EVALUATION RESULTS ({clips_evaluated} clips)")
    print("=" * 70)
    print(f"  Successes: {len(successes)}/{clips_evaluated} ({len(successes)/max(clips_evaluated,1)*100:.1f}%)")
    print(f"  Failures:  {len(failures)}/{clips_evaluated} ({len(failures)/max(clips_evaluated,1)*100:.1f}%)")
    print(f"  Avg progress: {sum(r['progress_pct'] for r in clip_results)/max(len(clip_results),1):.1f}%")

    print("\nFailures by termination reason:")
    print("-" * 50)
    for reason, items in sorted(failure_by_reason.items(), key=lambda x: -len(x[1])):
        avg_prog = sum(r["progress_pct"] for r in items) / len(items)
        print(f"  {reason:35s} {len(items):4d} clips  (avg progress: {avg_prog:.1f}%)")

    print("\nResults by motion type:")
    print("-" * 50)
    for desc, stats in sorted(by_desc.items(), key=lambda x: x[1]["successes"]/max(x[1]["count"],1)):
        sr = stats["successes"] / max(stats["count"], 1)
        avg_prog = stats["total_progress"] / max(stats["count"], 1)
        print(f"  {desc:35s} {stats['count']:3d} clips, {sr*100:5.1f}% success, avg progress {avg_prog:.1f}%")

    print("\n10 worst clips (lowest progress):")
    print("-" * 70)
    for r in sorted_by_progress[:10]:
        print(f"  clip {r['clip_id']:4d} | {r['clip_name']:30s} | {r['progress_pct']:5.1f}% | "
              f"term: {r['termination_reason']:20s} | desc: {r['clip_desc']}")

    print("=" * 70)

    # --- Append final summary line to JSONL ---
    summary = {
        "type": "summary",
        "num_clips_evaluated": clips_evaluated,
        "success_rate": len(successes) / max(clips_evaluated, 1),
        "avg_progress_pct": sum(r["progress_pct"] for r in clip_results) / max(len(clip_results), 1),
        "failures_by_reason": {k: len(v) for k, v in failure_by_reason.items()},
        "by_motion_type": by_desc,
        # "adaptive_sampling_state": sampling_state,
    }
    with open(results_path, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"\n[EVAL] Results saved to: {results_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
