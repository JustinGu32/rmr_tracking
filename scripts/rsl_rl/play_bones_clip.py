"""Play a trained bones policy on a single clip from a Zarr store.

Isolates one motion clip by clip_id or clip_name, sets the robot to the
clip start, and runs the policy for the clip duration. Supports --video.

Usage:
    python scripts/rsl_rl/play_bones_clip.py \
        --task=Bones-MultiClip-Compliance-G1-v0 \
        --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
        --wandb_path=robot-mcrobotface/multiclip_bones/run_id \
        --clip_id=42 \
        --num_envs=1 --headless --video

    # Or by name (substring match):
    python scripts/rsl_rl/play_bones_clip.py \
        --task=Bones-MultiClip-Compliance-G1-v0 \
        --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
        --wandb_path=robot-mcrobotface/multiclip_bones/run_id \
        --clip_name=walk_forward_normal_004__A006_M \
        --num_envs=1 --headless --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a bones policy on a single Zarr clip.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=None,
                    help="Length of the recorded video (in steps). Default: clip length.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--zarr_path", type=str, default=None, help="Path to Zarr store.")
parser.add_argument("--clip_id", type=int, default=None, help="Clip index in the Zarr store.")
parser.add_argument("--clip_name", type=str, default=None, help="Clip name (substring match).")
parser.add_argument("--decimation", type=int, default=None, help="Override env decimation.")
parser.add_argument("--video_dir", type=str, default=None, help="Directory to save video. Default: logs/rsl_rl/temp/videos/clip_play/")
parser.add_argument("--activation", type=str, default="elu", choices=["elu", "swish"],
                    help="Activation function for actor/critic networks (default: elu).")
parser.add_argument("--categories", type=str, default=None,
                    help="Comma-separated category names for PopArt tasks "
                         "(e.g. 'stand_up,walk'). Must match the training-time "
                         "categories so the env builds the same K-category command term.")
# PopArt opt-in. Mirrors scripts/rsl_rl/train_bones.py:69 — must match the
# training-time choice. Without this flag, vanilla MotionOnPolicyRunner is used
# even on the popart task (overriding the cfg's class_name default) and
# popart-specific policy fields are stripped so vanilla ActorCritic doesn't
# error on unknown kwargs. With --popart, PopArtMotionOnPolicyRunner is forced.
parser.add_argument("--popart", type=str, default="off", choices=["on", "off"],
                    help="Enable PopArt runner at playback (popart task only). "
                         "Must match the training-time --popart choice — pass "
                         "'on' if the checkpoint was trained with --popart=on, "
                         "'off' (default) otherwise.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Normalize --popart on/off → bool so downstream `if args_cli.popart` checks work unchanged.
args_cli.popart = (args_cli.popart == "on")

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import torch
import types
import zarr as _zarr

from rsl_rl.runners import OnPolicyRunner
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_from_euler_xyz, quat_inv, quat_mul, quat_apply, sample_uniform, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
import whole_body_tracking.tasks.tracking.mdp.commands as _tracking_cmds
import whole_body_tracking.tasks.popart.mdp.commands as _popart_cmds


def resolve_clip_id(zarr_path: str, clip_id: int | None, clip_name: str | None, exclude_objects: bool = True) -> tuple[int, str]:
    """Resolve a clip_id from either --clip_id or --clip_name.

    Returns (clip_id_in_filtered_dataset, clip_name_str).
    The clip_id corresponds to the index AFTER filtering (same as ZarrMotionLoader).
    """
    store = _zarr.open_group(zarr_path, mode="r")
    all_clip_start = store["clip_start_idx"][:]
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

    # Apply same filtering as ZarrMotionLoader
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
    else:
        valid_indices = list(range(total_clips_raw))

    filtered_names = [all_names[i] for i in valid_indices]

    if clip_id is not None:
        assert 0 <= clip_id < len(valid_indices), (
            f"clip_id {clip_id} out of range [0, {len(valid_indices)})")
        name = filtered_names[clip_id]
        print(f"[CLIP] Resolved clip_id={clip_id} → \"{name}\"")
        return clip_id, name

    if clip_name is not None:
        # Substring match
        matches = [(i, n) for i, n in enumerate(filtered_names) if clip_name.lower() in n.lower()]
        if len(matches) == 0:
            raise ValueError(f"No clips matching \"{clip_name}\" in {len(filtered_names)} filtered clips.")
        if len(matches) > 1:
            print(f"[CLIP] Multiple matches for \"{clip_name}\":")
            for i, n in matches[:20]:
                print(f"  clip_id={i}: {n}")
            if len(matches) > 20:
                print(f"  ... and {len(matches) - 20} more")
            print(f"[CLIP] Using first match: clip_id={matches[0][0]}")
        cid, name = matches[0]
        print(f"[CLIP] Resolved clip_name=\"{clip_name}\" → clip_id={cid} \"{name}\"")
        return cid, name

    raise ValueError("Must specify either --clip_id or --clip_name.")


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play a single clip with the trained policy."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.policy.activation = args_cli.activation
    env_cfg.scene.num_envs = args_cli.num_envs

    # PopArt is opt-in at play time too. Without --popart we force vanilla
    # MotionOnPolicyRunner (even on the popart task, overriding the cfg's
    # class_name default) and strip popart-specific policy fields so vanilla
    # ActorCritic doesn't error on unknown kwargs. With --popart, force
    # PopArtMotionOnPolicyRunner. Mirrors scripts/rsl_rl/train_bones.py:317-324.
    # The runner-dispatch block further down reads `agent_cfg.class_name`.
    # The strip is a no-op (guarded by hasattr) on non-popart tasks.
    if args_cli.popart:
        agent_cfg.class_name = "PopArtMotionOnPolicyRunner"
        print("[INFO] --popart enabled -> PopArtMotionOnPolicyRunner")
    else:
        agent_cfg.class_name = "MotionOnPolicyRunner"
        for k in ("num_categories", "popart_momentum", "category_obs_group"):
            if hasattr(agent_cfg.policy, k):
                delattr(agent_cfg.policy, k)
        print("[INFO] --popart not set -> MotionOnPolicyRunner (vanilla)")

    # --- Resolve clip ---
    zarr_path = args_cli.zarr_path
    if zarr_path is None:
        zarr_path = getattr(env_cfg.commands.motion, "zarr_path", "")
    assert zarr_path and os.path.isdir(zarr_path), f"--zarr_path required and must exist: {zarr_path}"
    env_cfg.commands.motion.zarr_path = zarr_path

    exclude_objects = getattr(env_cfg.commands.motion, "exclude_objects", True)
    clip_id, clip_name = resolve_clip_id(zarr_path, args_cli.clip_id, args_cli.clip_name, exclude_objects)

    # Get clip length from zarr for video_length default
    store = _zarr.open_group(zarr_path, mode="r")
    all_starts = store["clip_start_idx"][:]
    all_ends = store["clip_end_idx"][:]

    # Re-apply filtering to get the actual clip's frame range
    total_raw = len(all_starts)
    all_descs = [""] * total_raw
    if "content_props_desc" in store:
        raw_descs = store["content_props_desc"][:]
        for i in range(min(len(raw_descs), total_raw)):
            all_descs[i] = str(raw_descs[i])
    exclude_props = [
        "object manipulation", "wall", "chair", "obstacle",
        "edge", "safety pad", "railing", "box",
    ] if exclude_objects else None
    if exclude_props and "content_props_desc" in store:
        valid_indices = [i for i in range(total_raw)
                         if not any(ep.lower() in all_descs[i].strip().lower() for ep in exclude_props)]
    else:
        valid_indices = list(range(total_raw))

    raw_idx = valid_indices[clip_id]
    clip_length_frames = int(all_ends[raw_idx] - all_starts[raw_idx])
    print(f"[CLIP] \"{clip_name}\" — {clip_length_frames} frames")

    video_length = args_cli.video_length if args_cli.video_length is not None else clip_length_frames
    print(f"[CLIP] Will run for {video_length} steps")

    # --- Load model checkpoint ---
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    wandb_run_id = "local"

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api(timeout=60)
        if "model" in args_cli.wandb_path:
            run_path = "/".join(args_cli.wandb_path.split("/")[:-1])
        wandb_run = api.run(run_path)
        wandb_run_id = wandb_run.id

        files = [file.name for file in wandb_run.files() if "model" in file.name]
        if "model" in args_cli.wandb_path:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
        resume_path = f"./logs/rsl_rl/temp/{file}"
        print(f"[INFO] Loaded checkpoint: {run_path}/{file}")
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loaded checkpoint: {resume_path}")

    # --- Patch ZarrMotionLoader to load ONLY the target clip ---
    # Avoids loading the full zarr (~26 GB) when we only need one clip.
    # `MultiClipMotionCommand.__init__` looks up `ZarrMotionLoader` by name in
    # tracking.mdp.commands, so replacing the module attribute before
    # gym.make(...) causes the patched class to be instantiated instead.
    #
    # WARNING: this mirrors the state set by
    #   whole_body_tracking/tasks/tracking/mdp/commands.py :: ZarrMotionLoader.__init__
    # If that class adds attributes (or more tensors) that MultiClipMotionCommand
    # relies on, they must also be set here. Inheritance only saves us for the
    # @property accessors — the tensor state is rebuilt from scratch.
    _target_clip_id = clip_id

    class _SingleClipLoader(_tracking_cmds.ZarrMotionLoader):
        """Loads a single clip at local frames [0, L), rebasing clip_start/end_idx."""

        def __init__(self, zarr_path, body_indexes, device="cpu",
                     exclude_props=None, max_clips=None, include_keywords=None):
            # Skip super().__init__() deliberately — it eagerly loads the full zarr.
            store = _zarr.open(zarr_path, mode="r")
            self.fps = int(store["fps"][0])

            all_start = store["clip_start_idx"][:]
            all_end = store["clip_end_idx"][:]

            # Re-apply the same exclude_props filter used by the original loader,
            # so _target_clip_id indexes the same filtered list that resolve_clip_id used.
            if exclude_props and "content_props_desc" in store:
                descs = store["content_props_desc"][:]
                valid = [i for i in range(len(all_start))
                         if not any(ep.lower() in str(descs[i]).strip().lower() for ep in exclude_props)]
            else:
                valid = list(range(len(all_start)))

            raw = valid[_target_clip_id]
            s, e = int(all_start[raw]), int(all_end[raw])
            L = e - s

            # Load only this clip's frames; frame 0 = clip start.
            self.joint_pos = torch.tensor(store["joint_pos"][s:e], dtype=torch.float32, device=device)
            self.joint_vel = torch.tensor(store["joint_vel"][s:e], dtype=torch.float32, device=device)
            self._body_pos_w = torch.tensor(store["body_pos_w"][s:e], dtype=torch.float32, device=device)
            self._body_quat_w = torch.tensor(store["body_quat_w"][s:e], dtype=torch.float32, device=device)
            self._body_lin_vel_w = torch.tensor(store["body_lin_vel_w"][s:e], dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(store["body_ang_vel_w"][s:e], dtype=torch.float32, device=device)
            self._body_indexes = (body_indexes.to(device) if isinstance(body_indexes, torch.Tensor)
                                  else torch.tensor(body_indexes, dtype=torch.long, device=device))

            # Rebased metadata: loaded clip lives at local index 0, frames [0, L).
            self.clip_start_idx = torch.tensor([0], dtype=torch.long)
            self.clip_end_idx = torch.tensor([L], dtype=torch.long)
            self.num_clips = 1
            self.clip_lengths = self.clip_end_idx - self.clip_start_idx
            self.time_step_total = L
            # Needed by MultiClipMotionCommandPopArt.__init__, which iterates
            # self.motion.clip_names to build the clip→category lookup.
            _clip_name = str(store["clip_names"][raw]) if "clip_names" in store else f"clip_{raw}"
            self.clip_names = [_clip_name]

            print(f"[SingleClipLoader] Loaded clip_id={_target_clip_id} (raw zarr idx={raw}), "
                  f"name={_clip_name!r}, {L} frames @ {self.fps} fps, "
                  f"{self._body_pos_w.shape[1]} bodies, device={device}")

    # Patch BOTH copies of ZarrMotionLoader. The tracking task and the popart
    # task each have their own ZarrMotionLoader class in their own module, and
    # the popart task's MultiClipMotionCommand instantiates the popart copy
    # (via unqualified `ZarrMotionLoader(...)` in popart/mdp/commands.py:655).
    _tracking_cmds.ZarrMotionLoader = _SingleClipLoader
    _popart_cmds.ZarrMotionLoader = _SingleClipLoader

    # PopArt-task `categories` override (mirrors train_bones.py): single source
    # of truth driving num_categories + the priority categorizer + the include
    # filter. Must match training-time categories so the env builds the same
    # K-category command term that the checkpoint's K-head critic expects.
    if args_cli.categories is not None and hasattr(env_cfg.commands.motion, "categories"):
        cats = [s.strip() for s in args_cli.categories.split(",") if s.strip()]
        env_cfg.commands.motion.categories = cats
        print(f"[INFO] PopArt categories (priority order): {cats}")

    # Apply decimation override before computing fps so episode_length_s is sized correctly
    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

    # Set episode length long enough for the full clip (with margin)
    fps = 50.0  # default; overridden below if available
    if hasattr(env_cfg, 'decimation') and hasattr(env_cfg, 'sim') and hasattr(env_cfg.sim, 'dt'):
        fps = 1.0 / (env_cfg.sim.dt * env_cfg.decimation)
    env_cfg.episode_length_s = (clip_length_frames / fps) + 10.0
    # Disable perturbations
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "force_push_robot"):
            env_cfg.events.force_push_robot = None

    # Disable strict xy-only anchor position termination if present. Don't
    # touch `bad_anchor_pos` — that's treated as anchor-z in this project.
    if hasattr(env_cfg, "terminations"):
        for term_name in list(vars(env_cfg.terminations).keys()):
            term = getattr(env_cfg.terminations, term_name)
            func_name = getattr(getattr(term, "func", None), "__name__", "")
            if func_name == "bad_anchor_pos_x_y_only":
                print(f"[CLIP] Disabling xy termination: {term_name}")
                setattr(env_cfg.terminations, term_name, None)

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Wrap for video recording
    if args_cli.video:
        clip_subfolder = f"{clip_id}_{clip_name}"
        if args_cli.video_dir:
            video_folder = os.path.join(args_cli.video_dir, clip_subfolder)
        else:
            video_folder = os.path.join("logs", "rsl_rl", "temp", "videos", "clip_play", clip_subfolder)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": video_length,
            "disable_logger": True,
            "name_prefix": f"{'bones' if 'bones' in args_cli.task.lower() else 'tracking'}_{clip_id}_{clip_name}",
        }
        print(f"[INFO] Recording video to: {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env)

    # --- Load policy ---
    # Dispatch on agent_cfg.class_name so PopArt tasks get the PopArt runner
    # (which builds an ActorCriticPopArt with K-head critic + value_normalizer).
    # Mirrors _resolve_runner_class in scripts/rsl_rl/train_bones.py.
    _runner_class_name = getattr(agent_cfg, "class_name", None) or "MotionOnPolicyRunner"
    if _runner_class_name == "PopArtMotionOnPolicyRunner":
        from whole_body_tracking.utils.popart_runner import PopArtMotionOnPolicyRunner
        _RunnerCls = PopArtMotionOnPolicyRunner
    else:
        _RunnerCls = MotionOnPolicyRunner
    ppo_runner = _RunnerCls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    motion_cmd = raw_env.command_manager.get_term("motion")
    device = raw_env.device

    # --- Pin all envs to the target clip, starting at frame 0 ---
    def _pin_clip_resample(self, env_ids):
        """On every reset, pin back to the same clip at frame 0."""
        if len(env_ids) == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids

        # Loader has been patched to expose only the target clip at local
        # index 0 (see _SingleClipLoader above), so pin to index 0.
        self.clip_ids[env_ids_t] = 0
        self.clip_start[env_ids_t] = self.motion.clip_start_idx[0]
        self.clip_end[env_ids_t] = self.motion.clip_end_idx[0]
        self.time_steps[env_ids_t] = self.motion.clip_start_idx[0]

        self._cache_current_frames()

        # Write robot state to match reference
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

    # Replace _resample_command so every reset pins to our clip
    motion_cmd._resample_command = types.MethodType(_pin_clip_resample, motion_cmd)

    # --- Reset and warm up body_pos_relative_w ---
    obs, _ = env.reset()

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

    # --- Set up per-step diagnostics ---
    term_manager = raw_env.termination_manager
    term_names = term_manager._term_names
    ee_body_names = ["left_ankle_roll_link", "right_ankle_roll_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link"]

    from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes
    ee_body_indexes = _get_body_indexes(motion_cmd, ee_body_names)

    per_step_log = []
    terminated_at = None
    termination_reason = None
    termination_errors = None
    prev_step_data = None

    print(f"[CLIP] Playing clip_id={clip_id} \"{clip_name}\" for {video_length} steps...")

    # --- Main loop ---
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        eid = 0

        # Check terminations FIRST. env.step() has already reset terminated envs,
        # so motion_cmd state is now post-reset. The state that triggered the
        # termination is best approximated by prev_step_data (the previous
        # iteration's post-step reading, captured before this step's reset).
        if terminated_at is None:
            per_term = term_manager._term_dones[eid]
            fired = [term_names[i] for i in range(len(term_names)) if per_term[i]]
            if fired:
                terminated_at = timestep
                termination_reason = fired[0]
                termination_errors = prev_step_data
                print(f"[CLIP] Terminated at step {timestep}/{clip_length_frames}: {termination_reason}")
                if prev_step_data is not None:
                    print(f"  anchor_pos_err_xy={prev_step_data['anchor_pos_err_xy']:.4f}  "
                          f"anchor_pos_err_z={prev_step_data['anchor_pos_err_z']:.4f}")
                    for name, vals in prev_step_data["ee_z_errors"].items():
                        print(f"  {name}: z_error={vals['z_error']:.4f}")

        # Compute errors for env 0 (primary env for diagnostics)
        anchor_pos_err_xyz = (motion_cmd.anchor_pos_w[eid] - motion_cmd.robot_anchor_pos_w[eid]).cpu().tolist()
        anchor_pos_err_xy = float(torch.norm(motion_cmd.anchor_pos_w[eid, :2] - motion_cmd.robot_anchor_pos_w[eid, :2]).cpu())
        anchor_pos_err_z = float(torch.abs(motion_cmd.anchor_pos_w[eid, 2] - motion_cmd.robot_anchor_pos_w[eid, 2]).cpu())

        ee_z_errors = {}
        ee_z_ref = motion_cmd.body_pos_relative_w[eid, ee_body_indexes, -1].cpu()
        ee_z_robot = motion_cmd.robot_body_pos_w[eid, ee_body_indexes, -1].cpu()
        for i, name in enumerate(ee_body_names):
            ee_z_errors[name] = {
                "z_error": round(float(torch.abs(ee_z_ref[i] - ee_z_robot[i])), 4),
                "ref_z": round(float(ee_z_ref[i]), 4),
                "robot_z": round(float(ee_z_robot[i]), 4),
            }

        step_data = {
            "step": timestep,
            "anchor_pos_err_xy": round(anchor_pos_err_xy, 4),
            "anchor_pos_err_z": round(anchor_pos_err_z, 4),
            "anchor_pos_err_xyz": [round(v, 4) for v in anchor_pos_err_xyz],
            "ee_z_errors": ee_z_errors,
        }

        per_step_log.append(step_data)
        prev_step_data = step_data

        timestep += 1
        if timestep >= video_length:
            break

    # --- Save diagnostics ---
    import csv
    import json

    task_prefix = "bones" if "bones" in args_cli.task.lower() else "tracking"
    if args_cli.video:
        out_dir = video_folder
    else:
        out_dir = os.path.join("eval_results", "clip_play")
    os.makedirs(out_dir, exist_ok=True)
    base_name = f"{task_prefix}_{clip_id}_{clip_name}"

    # Summary JSON (small — no per-step data)
    summary = {
        "clip_id": clip_id,
        "clip_name": clip_name,
        "clip_length_frames": clip_length_frames,
        "task": args_cli.task,
        "wandb_path": args_cli.wandb_path,
        "steps_run": timestep,
        "terminated_at": terminated_at,
        "termination_reason": termination_reason,
        "termination_errors": termination_errors,
        "progress_pct": round((terminated_at if terminated_at else timestep) / max(clip_length_frames, 1) * 100, 1),
        "success": terminated_at is None or termination_reason == "time_out",
    }
    json_path = os.path.join(out_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-step CSV (easy to plot with pandas/matplotlib)
    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    csv_columns = ["step", "anchor_pos_err_xy", "anchor_pos_err_z"]
    csv_columns += [f"{name}_z_error" for name in ee_body_names]
    csv_columns += [f"{name}_ref_z" for name in ee_body_names]
    csv_columns += [f"{name}_robot_z" for name in ee_body_names]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in per_step_log:
            flat = {
                "step": row["step"],
                "anchor_pos_err_xy": row["anchor_pos_err_xy"],
                "anchor_pos_err_z": row["anchor_pos_err_z"],
            }
            for name in ee_body_names:
                flat[f"{name}_z_error"] = row["ee_z_errors"][name]["z_error"]
                flat[f"{name}_ref_z"] = row["ee_z_errors"][name]["ref_z"]
                flat[f"{name}_robot_z"] = row["ee_z_errors"][name]["robot_z"]
            writer.writerow(flat)

    print(f"[CLIP] Summary: {json_path}")
    print(f"[CLIP] Per-step errors: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
