"""Play a trained tracking policy (Tracking-Flat-G1-v0) from a wandb checkpoint.

Loads the motion file from the wandb run's used artifacts (same as training),
or from --registry_name if provided. Supports --video.

Usage:
    python scripts/rsl_rl/play_tracking_clip.py \
        --task=Tracking-Flat-G1-v0 \
        --wandb_path=robot-mcrobotface/multiclip_bones_popart/gerdxo4n \
        --num_envs=1 --headless --video

    # Override motion file:
    python scripts/rsl_rl/play_tracking_clip.py \
        --task=Tracking-Flat-G1-v0 \
        --wandb_path=robot-mcrobotface/multiclip_bones_popart/gerdxo4n \
        --registry_name=justingu-stanford-university-org/wandb-registry-Motions/crane_new:v0 \
        --num_envs=1 --headless --video
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a tracking policy loaded from a wandb checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record video.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Name of the task.")
parser.add_argument("--registry_name", type=str, default=None,
                    help="wandb registry path for motion artifact (e.g. org/registry/name:v0). "
                         "Auto-fetched from the run's used_artifacts if omitted.")
parser.add_argument("--video_dir", type=str, default=None,
                    help="Directory to save video. Default: logs/rsl_rl/temp/videos/tracking_play/")
parser.add_argument("--decimation", type=int, default=None, help="Override env decimation.")
parser.add_argument("--activation", type=str, default="swish", choices=["elu", "swish"],
                    help="Activation function for actor/critic networks (default: swish).")
parser.add_argument("--popart_head_mode", type=str, default=None, choices=["per_term", "grouped"],
                    help="Override PopArt head mode (auto-detected from checkpoint).")
parser.add_argument("--popart_group_preset", type=str, default=None,
                    help="Override PopArt group preset (auto-detected from checkpoint).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

os.environ["WBT_PPO_OUTPUT"] = "target"

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import pathlib
import torch

from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply, yaw_quat
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.tasks.bones.popart_reward_manager import install_bones_per_term_reward_manager
from whole_body_tracking.utils.bones_popart import BonesPopArtOnPolicyRunner
from whole_body_tracking.utils.hierarchical_popart import BonesCategoryRewardOnPolicyRunner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play the trained tracking policy."""
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    agent_cfg.policy.activation = args_cli.activation
    env_cfg.scene.num_envs = args_cli.num_envs

    # --- Load checkpoint from wandb ---
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    wandb_run = None

    if args_cli.wandb_path:
        import wandb

        run_path = args_cli.wandb_path
        api = wandb.Api()
        if "model" in run_path.split("/")[-1]:
            run_path = "/".join(run_path.split("/")[:-1])
        wandb_run = api.run(run_path)

        files = [f.name for f in wandb_run.files() if "model" in f.name and f.name.endswith(".pt")]
        if "model" in args_cli.wandb_path.split("/")[-1]:
            file = args_cli.wandb_path.split("/")[-1]
        else:
            if not files:
                raise RuntimeError(f"No model checkpoints found in wandb run: {run_path}")
            file = max(files, key=lambda x: int(x.split("_")[1].split(".")[0]))

        # Check local wandb cache first
        run_id = wandb_run.id
        local_wandb_dirs = sorted(pathlib.Path("./wandb").glob(f"run-*-{run_id}")) if pathlib.Path("./wandb").exists() else []
        local_run_dir = local_wandb_dirs[-1] / "files" if local_wandb_dirs else None
        resume_path = None

        if local_run_dir is not None and local_run_dir.exists():
            local_models = sorted(local_run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
            if local_models:
                local_path = local_run_dir / file if (local_run_dir / file).exists() else local_models[-1]
                print(f"[INFO] Loaded checkpoint from local cache: {local_path}")
                resume_path = str(local_path)

        if resume_path is None:
            wandb_run.file(str(file)).download("./logs/rsl_rl/temp", replace=True)
            resume_path = f"./logs/rsl_rl/temp/{file}"
            print(f"[INFO] Loaded checkpoint: {run_path}/{file}")
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO] Loaded checkpoint: {resume_path}")

    # --- Resolve motion file ---
    if args_cli.registry_name is not None:
        import wandb as _wandb
        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        api = _wandb.Api()
        artifact = api.artifact(registry_name)
        motion_path = str(pathlib.Path(artifact.download()) / "motion.npz")
        print(f"[INFO] Motion file from --registry_name: {motion_path}")
    elif wandb_run is not None:
        art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
        if art is None:
            raise RuntimeError(
                "No 'motions' artifact found in the wandb run. "
                "Pass --registry_name to specify the motion file explicitly."
            )
        motion_path = str(pathlib.Path(art.download()) / "motion.npz")
        print(f"[INFO] Motion file from used_artifacts: {motion_path} ({art.name})")
    else:
        raise RuntimeError("Either --wandb_path or --registry_name must be provided to locate the motion file.")

    if hasattr(env_cfg.commands.motion, "motion_files"):
        env_cfg.commands.motion.motion_files = [motion_path]
    else:
        env_cfg.commands.motion.motion_file = motion_path

    if args_cli.decimation is not None:
        env_cfg.decimation = args_cli.decimation

    # Disable perturbations for clean playback
    if hasattr(env_cfg, "events"):
        if hasattr(env_cfg.events, "push_robot"):
            env_cfg.events.push_robot = None
        if hasattr(env_cfg.events, "force_push_robot"):
            env_cfg.events.force_push_robot = None

    # --- Create environment ---
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.video:
        run_label = wandb_run.id if wandb_run is not None else "local"
        if args_cli.video_dir:
            video_folder = args_cli.video_dir
        else:
            video_folder = os.path.join("logs", "rsl_rl", "temp", "videos", "tracking_play", run_label)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
            "name_prefix": f"tracking_{run_label}",
        }
        print(f"[INFO] Recording video to: {video_folder}")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # --- Auto-detect PopArt runner from checkpoint ---
    _ckpt = torch.load(resume_path, map_location="cpu")
    _msd = _ckpt.get("model_state_dict", _ckpt)
    use_popart = any(k.startswith("critic_trunk.") for k in _msd)
    _mean = _msd.get("value_normalizer.mean")
    use_hierarchical = use_popart and _mean is not None and _mean.ndim == 2

    if use_hierarchical:
        print(f"[INFO] Detected hierarchical PopArt checkpoint ({_mean.shape[0]} categories × {_mean.shape[1]} heads).")
        agent_cfg.algorithm.popart_hierarchical = True
        agent_cfg.algorithm.popart_num_categories = int(_mean.shape[0])
        if args_cli.popart_head_mode is not None:
            agent_cfg.algorithm.popart_head_mode = args_cli.popart_head_mode
        if args_cli.popart_group_preset is not None:
            agent_cfg.algorithm.popart_group_preset = args_cli.popart_group_preset
        install_bones_per_term_reward_manager(env)
    elif use_popart:
        print("[INFO] Detected PopArt checkpoint.")
        agent_cfg.algorithm.use_popart_multihead = True
        install_bones_per_term_reward_manager(env)

    env = RslRlVecEnvWrapper(env)

    # --- Load policy ---
    if use_hierarchical:
        runner_cls = BonesCategoryRewardOnPolicyRunner
    elif use_popart:
        runner_cls = BonesPopArtOnPolicyRunner
    else:
        runner_cls = MotionOnPolicyRunner
    ppo_runner = runner_cls(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    raw_env = env.unwrapped
    motion_cmd = raw_env.command_manager.get_term("motion")

    ee_body_names = ["left_ankle_roll_link", "right_ankle_roll_link",
                     "left_wrist_yaw_link", "right_wrist_yaw_link"]

    from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes
    ee_body_indexes = _get_body_indexes(motion_cmd, ee_body_names)

    # --- Reset ---
    obs, _ = env.reset()

    # Warm up body_pos_relative_w (only available on ZarrMotionCommand, not all MotionCommand types)
    if hasattr(motion_cmd, "_cache_current_frames"):
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

    per_step_log = []

    print(f"[INFO] Playing for {args_cli.video_length} steps...")

    # --- Main loop ---
    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        eid = 0
        anchor_pos_err_xy = float(torch.norm(motion_cmd.anchor_pos_w[eid, :2] - motion_cmd.robot_anchor_pos_w[eid, :2]).cpu())
        anchor_pos_err_z = float(torch.abs(motion_cmd.anchor_pos_w[eid, 2] - motion_cmd.robot_anchor_pos_w[eid, 2]).cpu())
        anchor_pos_err_xyz = (motion_cmd.anchor_pos_w[eid] - motion_cmd.robot_anchor_pos_w[eid]).cpu().tolist()

        ee_z_errors = {}
        ee_z_ref = motion_cmd.body_pos_relative_w[eid, ee_body_indexes, -1].cpu()
        ee_z_robot = motion_cmd.robot_body_pos_w[eid, ee_body_indexes, -1].cpu()
        for i, name in enumerate(ee_body_names):
            ee_z_errors[name] = {
                "z_error": round(float(torch.abs(ee_z_ref[i] - ee_z_robot[i])), 4),
                "ref_z": round(float(ee_z_ref[i]), 4),
                "robot_z": round(float(ee_z_robot[i]), 4),
            }

        per_step_log.append({
            "step": timestep,
            "anchor_pos_err_xy": round(anchor_pos_err_xy, 4),
            "anchor_pos_err_z": round(anchor_pos_err_z, 4),
            "anchor_pos_err_xyz": [round(v, 4) for v in anchor_pos_err_xyz],
            "ee_z_errors": ee_z_errors,
        })

        timestep += 1
        if timestep >= args_cli.video_length:
            break

    # --- Save diagnostics ---
    import csv, json

    out_dir = video_folder if args_cli.video else os.path.join("eval_results", "tracking_play")
    os.makedirs(out_dir, exist_ok=True)
    run_label = wandb_run.id if wandb_run is not None else "local"
    base_name = f"tracking_{run_label}"

    json_path = os.path.join(out_dir, f"{base_name}.json")
    with open(json_path, "w") as f:
        json.dump({
            "task": args_cli.task,
            "wandb_path": args_cli.wandb_path,
            "steps_run": timestep,
            "video_length": args_cli.video_length,
        }, f, indent=2)

    csv_path = os.path.join(out_dir, f"{base_name}.csv")
    csv_columns = ["step", "anchor_pos_err_xy", "anchor_pos_err_z"]
    csv_columns += [f"{name}_z_error" for name in ee_body_names]
    csv_columns += [f"{name}_ref_z" for name in ee_body_names]
    csv_columns += [f"{name}_robot_z" for name in ee_body_names]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for row in per_step_log:
            flat = {"step": row["step"], "anchor_pos_err_xy": row["anchor_pos_err_xy"], "anchor_pos_err_z": row["anchor_pos_err_z"]}
            for name in ee_body_names:
                flat[f"{name}_z_error"] = row["ee_z_errors"][name]["z_error"]
                flat[f"{name}_ref_z"] = row["ee_z_errors"][name]["ref_z"]
                flat[f"{name}_robot_z"] = row["ee_z_errors"][name]["robot_z"]
            writer.writerow(flat)

    print(f"[INFO] Summary: {json_path}")
    print(f"[INFO] Per-step errors: {csv_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
