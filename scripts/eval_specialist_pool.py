"""Evaluate a trained baseline policy across every clip in a Zarr store.

For each clip, runs N deterministic passes from a uniform start frame (or
frame 0). A clip is "failed" if the policy terminates early (non-time_out)
in at least one pass.

Outputs:
    <output_dir>/failed_clip_ids.json   list[str] of failed clip names
    <output_dir>/eval_summary.csv       one row per (clip, pass)

Usage:
    python scripts/eval_specialist_pool.py \\
        --task=Popart-Flat-G1-Play-v0 \\
        --wandb_path=robot-mcrobotface/balanced_sampling/<baseline_id> \\
        --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_standup_walk_jump_all.zarr \\
        --categories stand_up,walk,jump --num_passes 3 \\
        --start_frame_mode uniform --decimation 6 --activation swish \\
        --output_dir eval_results/balanced_ia5mxune/
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.insert(0, "scripts/rsl_rl")
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Multi-pass deterministic eval to find failed clips.")
parser.add_argument("--num_envs", type=int, default=4096,
                    help="Number of envs. With >1 envs, evaluates many (clip_id, pass_idx) units "
                         "in parallel — each env is pinned to a different clip via a per-env "
                         "target tensor. Auto-reset on early termination is ignored (we record "
                         "the first failure per env). Set to 1 for the original sequential eval.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr store.")
parser.add_argument("--decimation", type=int, default=None, help="Override env decimation.")
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"])
parser.add_argument("--categories", type=str, default=None,
                    help="Comma-separated category names (popart task). Must match training-time.")
parser.add_argument("--popart", type=str, default="off", choices=["on", "off"],
                    help="Enable PopArt runner — must match training-time choice.")
# Eval-specific
parser.add_argument("--num_passes", type=int, default=3,
                    help="Number of deterministic passes per clip (default: 3).")
parser.add_argument("--start_frame_mode", type=str, default="uniform", choices=["zero", "uniform"],
                    help="Per-pass start frame: 'zero' = clip_start every pass, 'uniform' = random in [clip_start, clip_end).")
parser.add_argument("--max_steps_margin", type=int, default=30,
                    help="Extra policy steps beyond clip length before declaring success (default: 30).")
parser.add_argument("--max_steps_per_pass", type=int, default=None,
                    help="HARD cap on steps per pass, regardless of clip_length. If the policy "
                         "survives this many steps without terminating, mark as success (no_term). "
                         "Catches 'failure happens early or not at all' cases efficiently on long "
                         "clips. Default: no cap (use clip_length + max_steps_margin).")
parser.add_argument("--clip_sample_size", type=int, default=None,
                    help="If set, randomly sample this many clip IDs (with seed) and only eval those. "
                         "Useful when the full dataset is too large for a single SLURM job.")
parser.add_argument("--clip_sample_seed", type=int, default=0,
                    help="RNG seed for --clip_sample_size (default: 0). Reproducible subsampling.")
parser.add_argument("--output_dir", type=str, required=True,
                    help="Directory for failed_clip_ids.json and eval_summary.csv.")
parser.add_argument("--include_clip_names_file", type=str, default=None,
                    help="Optional JSON list of clip names to restrict eval to "
                         "(intersection with --categories). Use this to re-eval a specialist "
                         "against the failed-clip set.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Normalize --popart on/off → bool.
args_cli.popart = (args_cli.popart == "on")

# Force headless for eval (no rendering needed).
if not getattr(args_cli, "headless", False):
    args_cli.headless = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv
import gymnasium as gym
import json
import os
import torch
import types

from rsl_rl.runners import OnPolicyRunner  # noqa: F401 — keep for symmetry with play_bones_clip
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401


def _pin_clip_resample_parametric(self, env_ids):
    """Reset hook that pins each resetting env to its assigned clip.

    Reads per-env clip assignment from `self._pin_target_clip_ids` (long
    tensor, shape [num_envs]). Start frame controlled by
    `self._pin_start_mode`:
        "zero"    -> always clip_start_i
        "uniform" -> random in [clip_start_i, clip_end_i) per env

    Vectorized over env_ids so different envs can run different clips in
    parallel during batched eval.
    """
    # # === OLD (single-target) implementation, kept for reference. ===
    # # Pinned ALL resetting envs to `self._pin_target_clip_id` (scalar). Replaced
    # # by per-env `_pin_target_clip_ids` (tensor) for batched eval.
    # if len(env_ids) == 0:
    #     return
    # env_ids_t = torch.as_tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
    # target = int(self._pin_target_clip_id)
    # clip_start = int(self.motion.clip_start_idx[target].item())
    # clip_end = int(self.motion.clip_end_idx[target].item())
    # clip_len = max(clip_end - clip_start, 1)
    # self.clip_ids[env_ids_t] = target
    # self.clip_start[env_ids_t] = self.motion.clip_start_idx[target]
    # self.clip_end[env_ids_t] = self.motion.clip_end_idx[target]
    # if getattr(self, "_pin_start_mode", "zero") == "uniform":
    #     offsets = torch.randint(0, clip_len, (len(env_ids_t),), device=self.device, dtype=torch.long)
    #     self.time_steps[env_ids_t] = self.motion.clip_start_idx[target] + offsets
    # else:
    #     self.time_steps[env_ids_t] = self.motion.clip_start_idx[target]
    # if hasattr(self, "clip_to_category") and hasattr(self, "category_idx"):
    #     self.category_idx[env_ids_t] = self.clip_to_category[target]
    # self._cache_current_frames()

    if len(env_ids) == 0:
        return
    env_ids_t = torch.as_tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

    # Per-env clip assignment.
    targets = self._pin_target_clip_ids[env_ids_t]  # [n]
    clip_starts = self.motion.clip_start_idx[targets]  # [n]
    clip_ends = self.motion.clip_end_idx[targets]  # [n]
    clip_lens = (clip_ends - clip_starts).clamp_min(1)  # [n]

    self.clip_ids[env_ids_t] = targets
    self.clip_start[env_ids_t] = clip_starts
    self.clip_end[env_ids_t] = clip_ends

    if getattr(self, "_pin_start_mode", "zero") == "uniform":
        # Per-env random offset in [0, clip_len_i).
        rand_unit = torch.rand(len(env_ids_t), device=self.device)
        offsets = (rand_unit * clip_lens.float()).long()
        self.time_steps[env_ids_t] = clip_starts + offsets
    else:
        self.time_steps[env_ids_t] = clip_starts

    # Keep category_idx in sync for PopArt tasks (no-op on non-popart).
    if hasattr(self, "clip_to_category") and hasattr(self, "category_idx"):
        self.category_idx[env_ids_t] = self.clip_to_category[targets]

    self._cache_current_frames()

    # Write robot state to match reference (same pattern as play_bones_clip).
    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids_t]
    joint_pos[env_ids_t] = torch.clip(
        joint_pos[env_ids_t], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(joint_pos[env_ids_t], joint_vel[env_ids_t], env_ids=env_ids_t)
    self.robot.write_root_state_to_sim(
        torch.cat([root_pos[env_ids_t], root_ori[env_ids_t],
                   root_lin_vel[env_ids_t], root_ang_vel[env_ids_t]], dim=-1),
        env_ids=env_ids_t,
    )


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.policy.activation = args_cli.activation
    env_cfg.scene.num_envs = args_cli.num_envs

    # PopArt opt-in (mirror play_bones_clip / train_bones).
    if args_cli.popart:
        agent_cfg.class_name = "PopArtMotionOnPolicyRunner"
    else:
        agent_cfg.class_name = "MotionOnPolicyRunner"
        for k in ("num_categories", "popart_momentum", "category_obs_group"):
            if hasattr(agent_cfg.policy, k):
                delattr(agent_cfg.policy, k)

    # Configure zarr and category filter on the cfg before gym.make.
    assert args_cli.zarr_path is not None, "--zarr_path required"
    assert os.path.isdir(args_cli.zarr_path), f"Zarr path not found: {args_cli.zarr_path}"
    env_cfg.commands.motion.zarr_path = args_cli.zarr_path

    if args_cli.categories is not None and hasattr(env_cfg.commands.motion, "categories"):
        cats = [s.strip() for s in args_cli.categories.split(",") if s.strip()]
        env_cfg.commands.motion.categories = cats
        print(f"[INFO] PopArt categories (priority order): {cats}")

    # Optional clip-name filter (used when re-evaluating a specialist on its
    # training subset). Plumbed via the same cfg field that train_bones uses.
    if args_cli.include_clip_names_file is not None and hasattr(env_cfg.commands.motion, "include_clip_names"):
        with open(args_cli.include_clip_names_file, "r") as f:
            names = json.load(f)
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError("--include_clip_names_file must be a flat JSON list of strings.")
        env_cfg.commands.motion.include_clip_names = names
        print(f"[INFO] Restricting eval to {len(names)} clips from {args_cli.include_clip_names_file}")

    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

    # Make episode_length_s large — we'll cap manually per-clip below. This
    # keeps the time_out termination from firing inside short clips.
    env_cfg.episode_length_s = 10_000.0

    # Disable random perturbations so eval is deterministic.
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "force_push_robot"):
            env_cfg.events.force_push_robot = None

    # --- Load checkpoint from wandb ---
    assert args_cli.wandb_path, "--wandb_path required"
    import wandb
    run_path = args_cli.wandb_path
    api = wandb.Api(timeout=60)
    if "model" in args_cli.wandb_path:
        run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
    wandb_run = api.run(run_path)
    files = [file.name for file in wandb_run.files() if "model" in file.name]
    if "model" in args_cli.wandb_path:
        file = args_cli.wandb_path.split("/")[-1]
    else:
        file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))
    wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
    resume_path = f"./logs/rsl_rl/temp/{file}"
    print(f"[INFO] Loaded checkpoint: {run_path}/{file}")

    # --- Build env ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env)

    # --- Build runner and load weights ---
    _runner_class_name = getattr(agent_cfg, "class_name", None) or "MotionOnPolicyRunner"
    if _runner_class_name == "PopArtMotionOnPolicyRunner":
        from whole_body_tracking.utils.popart_runner import PopArtMotionOnPolicyRunner
        _RunnerCls = PopArtMotionOnPolicyRunner
    else:
        _RunnerCls = MotionOnPolicyRunner
    runner = _RunnerCls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    # # OLD: runner.load(resume_path)
    # # DAgger-saved checkpoints don't carry optimizer_state_dict or `infos`
    # # (we never resume DAgger). rsl_rl's runner.load reads both unconditionally
    # # for `infos` and conditionally for the optimizer. Inject empty `infos`
    # # if absent and skip optimizer loading — both are inference-irrelevant.
    _loaded = torch.load(resume_path, weights_only=False, map_location="cpu")
    if "infos" not in _loaded:
        _loaded["infos"] = {}
        torch.save(_loaded, resume_path)
        print(f"[INFO] injected missing 'infos' key into {resume_path}")
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    motion_cmd = raw_env.command_manager.get_term("motion")

    # --- Install parametric pin override ---
    motion_cmd._resample_command = types.MethodType(_pin_clip_resample_parametric, motion_cmd)
    # # OLD single-target attribute (replaced by per-env tensor below).
    # motion_cmd._pin_target_clip_id = 0
    motion_cmd._pin_target_clip_ids = torch.zeros(
        args_cli.num_envs, dtype=torch.long, device=raw_env.device
    )
    motion_cmd._pin_start_mode = args_cli.start_frame_mode

    # Pull per-clip metadata.
    clip_names = list(motion_cmd.motion.clip_names)
    num_clips = motion_cmd.motion.num_clips
    clip_start_arr = motion_cmd.motion.clip_start_idx.cpu().tolist()
    clip_end_arr = motion_cmd.motion.clip_end_idx.cpu().tolist()
    # Optional clip subsampling — for huge datasets where evaluating every clip
    # would exceed the SLURM time limit. Random sample with fixed seed for
    # reproducibility.
    if args_cli.clip_sample_size is not None and args_cli.clip_sample_size < num_clips:
        import random as _random
        _rng = _random.Random(args_cli.clip_sample_seed)
        clip_id_order = _rng.sample(range(num_clips), args_cli.clip_sample_size)
        print(f"[EVAL] subsampled {args_cli.clip_sample_size}/{num_clips} clips "
              f"(seed={args_cli.clip_sample_seed})", flush=True)
    else:
        clip_id_order = list(range(num_clips))

    print(f"[EVAL] {len(clip_id_order)} clips to eval, {args_cli.num_passes} passes each, "
          f"start_frame_mode={args_cli.start_frame_mode}, "
          f"max_steps_per_pass={args_cli.max_steps_per_pass}", flush=True)

    term_manager = raw_env.termination_manager
    term_names = term_manager._term_names

    # --- Open output files NOW so partial results are visible if the job
    # dies mid-eval. CSV gets a flushed row per (clip, pass); JSON is rewritten
    # after every clip so failed_clip_ids.json always reflects current state.
    os.makedirs(args_cli.output_dir, exist_ok=True)
    json_path = os.path.join(args_cli.output_dir, "failed_clip_ids.json")
    csv_path = os.path.join(args_cli.output_dir, "eval_summary.csv")
    csv_fieldnames = ["clip_id", "clip_name", "pass_idx", "terminated",
                      "terminal_step", "terminal_reason", "clip_length"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fieldnames)
    csv_writer.writeheader()
    csv_file.flush()
    # Seed the JSON with an empty list so downstream code sees a valid file.
    with open(json_path, "w") as _jf:
        json.dump([], _jf, indent=2)
    print(f"[EVAL] live results -> {csv_path}", flush=True)
    print(f"[EVAL] live failed list -> {json_path}", flush=True)

    # # === OLD sequential (num_envs=1) main loop, kept for reference. ===
    # # Outer loop over clips, inner loop over passes, deepest loop over steps.
    # # Replaced by the batched implementation below which packs num_envs units
    # # (clip_id, pass_idx) into each batch and runs them in parallel.
    # results: list[dict] = []
    # failed_clip_names: list[str] = []
    # for loop_idx, clip_id in enumerate(clip_id_order):
    #     clip_name = clip_names[clip_id]
    #     clip_len = int(clip_end_arr[clip_id] - clip_start_arr[clip_id])
    #     max_steps = clip_len + args_cli.max_steps_margin
    #     if args_cli.max_steps_per_pass is not None:
    #         max_steps = min(max_steps, args_cli.max_steps_per_pass)
    #     clip_failed = False
    #     for pass_idx in range(args_cli.num_passes):
    #         motion_cmd._pin_target_clip_id = clip_id
    #         motion_cmd._pin_start_mode = args_cli.start_frame_mode
    #         obs, _ = env.reset()
    #         terminated_at = None
    #         terminal_reason = None
    #         for step in range(max_steps):
    #             with torch.no_grad():
    #                 actions = policy(obs)
    #                 obs, _, _, _ = env.step(actions)
    #             per_term = term_manager._term_dones[0]
    #             fired = [term_names[i] for i in range(len(term_names)) if per_term[i]]
    #             if fired:
    #                 non_timeout = [t for t in fired if t != "time_out"]
    #                 if non_timeout:
    #                     terminated_at = step
    #                     terminal_reason = non_timeout[0]
    #                 else:
    #                     terminal_reason = "time_out"
    #                 break
    #         terminated = (terminated_at is not None) and (terminal_reason != "time_out")
    #         row = {...}
    #         results.append(row); csv_writer.writerow(row); csv_file.flush()
    #         if terminated: clip_failed = True
    #     if clip_failed: failed_clip_names.append(clip_name)
    #     with open(json_path, "w") as _jf: json.dump(failed_clip_names, _jf, indent=2)
    # csv_file.close()

    # --- Batched main eval loop ---
    # Pack (clip_id, pass_idx) units into batches of num_envs and run them in
    # parallel. Each env tracks its own first-failure step + reason. Auto-resets
    # after termination are ignored (we stop reading those envs).
    device = raw_env.device
    num_envs = args_cli.num_envs
    num_terms = len(term_names)
    # Indices of non-timeout terms (used for failure detection in priority
    # order matching the original code's preference for the first fired term).
    non_to_term_idxs = [i for i, n in enumerate(term_names) if n != "time_out"]
    if not non_to_term_idxs:
        raise RuntimeError("No non-timeout terminations registered — eval can't detect failures.")

    # Flatten clips × passes into (clip_id, pass_idx) units.
    units: list[tuple[int, int]] = [
        (cid, p) for cid in clip_id_order for p in range(args_cli.num_passes)
    ]
    num_units = len(units)
    num_batches = (num_units + num_envs - 1) // num_envs

    print(f"[EVAL] {len(clip_id_order)} clips × {args_cli.num_passes} passes = {num_units} units, "
          f"batched into {num_batches} batches of <= {num_envs} envs", flush=True)

    # Per-clip aggregation across all completed units. clip_failed_count[cid] is
    # the number of passes seen so far that terminated. Used to incrementally
    # rebuild failed_clip_ids.json after each batch.
    clip_failed_count: dict[int, int] = {}
    results_total = 0

    for batch_idx in range(num_batches):
        batch_start = batch_idx * num_envs
        batch = units[batch_start:batch_start + num_envs]
        actual_batch = len(batch)

        # Per-env clip assignment. Filler slots (i >= actual_batch) get clip 0;
        # their results are discarded and they're marked done from the start.
        batch_clip_ids_cpu = [c for c, _ in batch] + [0] * (num_envs - actual_batch)
        batch_clip_ids = torch.tensor(batch_clip_ids_cpu, dtype=torch.long, device=device)
        motion_cmd._pin_target_clip_ids = batch_clip_ids
        motion_cmd._pin_start_mode = args_cli.start_frame_mode

        # Per-env step budget = clip_len + margin, optionally capped. Filler
        # envs get budget 0 (immediately marked done).
        env_max_steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        for i, (cid, _) in enumerate(batch):
            clen = int(clip_end_arr[cid] - clip_start_arr[cid])
            budget = clen + args_cli.max_steps_margin
            if args_cli.max_steps_per_pass is not None:
                budget = min(budget, args_cli.max_steps_per_pass)
            env_max_steps[i] = budget
        max_steps_batch = int(env_max_steps.max().item())

        # Reset env — _pin_clip_resample_parametric reads batch_clip_ids per env.
        obs, _ = env.reset()

        # Per-env termination tracking.
        env_term_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        env_term_reason_idx = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        env_recorded = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Filler envs start as "recorded" so they're ignored everywhere.
        if actual_batch < num_envs:
            env_recorded[actual_batch:] = True

        for step in range(max_steps_batch):
            with torch.no_grad():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

            per_env_term = term_manager._term_dones  # [num_envs, num_terms] bool

            # Mark envs that exhausted their per-env budget as "no_term" success
            # (env_recorded=True, reason left at -1).
            expired = (step >= env_max_steps - 1) & ~env_recorded
            env_recorded |= expired

            # Detect new non-timeout failures.
            non_to_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
            for t_idx in non_to_term_idxs:
                non_to_mask |= per_env_term[:, t_idx]
            newly_failed = non_to_mask & ~env_recorded
            if newly_failed.any():
                env_term_step[newly_failed] = step
                # Assign reason in term_names priority order (first fired wins),
                # matching the original sequential code.
                for t_idx in non_to_term_idxs:
                    this_t = (
                        per_env_term[:, t_idx]
                        & newly_failed
                        & (env_term_reason_idx == -1)
                    )
                    if this_t.any():
                        env_term_reason_idx[this_t] = t_idx
                env_recorded |= newly_failed

            if env_recorded.all():
                break

        # Materialize results for this batch.
        env_term_step_cpu = env_term_step.cpu().tolist()
        env_term_reason_idx_cpu = env_term_reason_idx.cpu().tolist()
        batch_term_count = 0
        for i, (cid, pidx) in enumerate(batch):
            terminated = env_term_reason_idx_cpu[i] != -1
            if terminated:
                step_val = env_term_step_cpu[i]
                reason = term_names[env_term_reason_idx_cpu[i]]
                batch_term_count += 1
            else:
                step_val = int(env_max_steps[i].item())
                reason = "no_term"
            clen = int(clip_end_arr[cid] - clip_start_arr[cid])
            row = {
                "clip_id": cid,
                "clip_name": clip_names[cid],
                "pass_idx": pidx,
                "terminated": int(terminated),
                "terminal_step": step_val,
                "terminal_reason": reason,
                "clip_length": clen,
            }
            csv_writer.writerow(row)
            if terminated:
                clip_failed_count[cid] = clip_failed_count.get(cid, 0) + 1
            results_total += 1
        csv_file.flush()

        # Rebuild failed_clip_ids.json from the running aggregate.
        failed_clip_names_now = [
            clip_names[c] for c in clip_id_order if clip_failed_count.get(c, 0) > 0
        ]
        with open(json_path, "w") as _jf:
            json.dump(failed_clip_names_now, _jf, indent=2)

        print(
            f"  [batch {batch_idx+1}/{num_batches}] units={actual_batch} "
            f"max_steps={max_steps_batch} terminated={batch_term_count}/{actual_batch} "
            f"cum_failed_clips={len(failed_clip_names_now)} cum_units={results_total}/{num_units}",
            flush=True,
        )

    csv_file.close()
    failed_clip_names = [
        clip_names[c] for c in clip_id_order if clip_failed_count.get(c, 0) > 0
    ]

    print("\n[EVAL DONE]", flush=True)
    print(f"  Total clips:   {num_clips}", flush=True)
    print(f"  Failed clips:  {len(failed_clip_names)}  (≥ 1 termination in {args_cli.num_passes} passes)", flush=True)
    print(f"  Output:        {args_cli.output_dir}", flush=True)
    print(f"    {json_path}", flush=True)
    print(f"    {csv_path}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
