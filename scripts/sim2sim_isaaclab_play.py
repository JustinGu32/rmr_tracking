"""Script to play a diffusion policy checkpoint in Isaac Lab.

Uses the same structure as play.py but replaces the RL policy with DiffusionAgentIsaac.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from pathlib import Path

# Add TML-BeyondMimic for diffusion policy (sibling of rmr_tracking)
_SCRIPT_DIR = Path(__file__).resolve().parent
_RMR_ROOT = _SCRIPT_DIR.parent.parent
_TML_ROOT = _RMR_ROOT.parent / "TML-BeyondMimic"
if _TML_ROOT.exists():
    sys.path.insert(0, str(_TML_ROOT))

from isaaclab.app import AppLauncher

# add argparse arguments (same as play.py)
parser = argparse.ArgumentParser(description="Play diffusion policy in Isaac Lab (replaces RL agent).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Name of the task.")
parser.add_argument("--motion_file", type=str, default='/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz', help="Path to the motion file.")
# Diffusion-specific (one of checkpoint or wandb_path required)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local diffusion checkpoint (.pt)")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path (e.g. user/project/run_id)")
parser.add_argument("--wandb_file", type=str, default="latest.ckpt", help="Checkpoint filename in wandb")
parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic sampling")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import numpy as np
import torch

import gymnasium as gym
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401

# Diffusion policy (from TML-BeyondMimic)
from diffusion_policy.inference.diffusion_agent import DiffusionAgentIsaac

# Isaac action scale and default pose (policy outputs normalized; target = action * scale + default)
ACTION_SCALE_ISAAC = np.array([
    0.548, 0.548, 0.548, 0.351, 0.351, 0.439, 0.548, 0.548, 0.439,
    0.351, 0.351, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439,
    0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.439, 0.075, 0.075, 0.075, 0.075,
], dtype=np.float32)
DEFAULT_POSE_ISAAC = np.array([
    -0.312, -0.312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.669, 0.669, 0.2, 0.2, -0.363, -0.363, 0.2, -0.2, 0.0, 0.0,
    0.0, 0.0, 0.6, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
], dtype=np.float32)


def _isaac_action_to_env(action_isaac: np.ndarray, env, env_id: int = 0) -> torch.Tensor:
    """Convert diffusion policy output (Isaac order, unnormalized target positions) to env action.

    Policy returns target = action * scale + default. Env expects: joint target = offset + scale * action_env,
    so action_env = (target - offset) / scale.
    """
    target_isaac = action_isaac * ACTION_SCALE_ISAAC + DEFAULT_POSE_ISAAC
    robot = env.unwrapped.scene["robot"]
    target_t = torch.tensor(target_isaac, device=robot.device, dtype=torch.float32)
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    scale = action_term._scale[env_id]
    offset = action_term._offset[env_id]
    action_env = (target_t - offset) / scale
    return action_env.unsqueeze(0)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg=None):
    """Play with diffusion policy (replaces RL agent in play.py)."""
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # Validate diffusion checkpoint args
    if args_cli.checkpoint is None and args_cli.wandb_path is None:
        print("[ERROR] Diffusion policy requires --checkpoint or --wandb_path")
        sys.exit(1)
    if args_cli.checkpoint is not None and args_cli.wandb_path is not None:
        print("[ERROR] Specify only one of --checkpoint or --wandb_path")
        sys.exit(1)

    if args_cli.motion_file is not None:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # create isaac environment (same as play.py)
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording (same as play.py)
    if args_cli.video:
        video_folder = os.path.join("videos", "play")
        if args_cli.checkpoint:
            video_folder = os.path.join(os.path.dirname(args_cli.checkpoint), "videos", "play")
        os.makedirs(video_folder, exist_ok=True)
        video_kwargs = {
            "video_folder": video_folder,
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl (same as play.py)
    env = RslRlVecEnvWrapper(env)

    # Load diffusion policy instead of RL policy
    device = getattr(env.unwrapped, "device", None) or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] Loading diffusion policy (DiffusionAgentIsaac)...")
    if args_cli.checkpoint:
        policy = DiffusionAgentIsaac(
            checkpoint_path=args_cli.checkpoint,
            device=device,
            compile=False,
            warmup=False,
            deterministic=args_cli.deterministic,
        )
    else:
        policy = DiffusionAgentIsaac(
            wandb_path=args_cli.wandb_path,
            checkpoint_file=args_cli.wandb_file,
            device=device,
            compile=False,
            warmup=False,
            deterministic=args_cli.deterministic,
        )
    print("[INFO] Diffusion policy loaded.")
    policy.reset()

    # reset environment (same as play.py)
    obs, _ = env.reset()
    timestep = 0
    env_id = 0
    last_action_isaac = None

    # simulate environment (same loop structure as play.py)
    while simulation_app.is_running():
        with torch.inference_mode():
            # Extract diffusion state from obs['diffusion_collect']
            dc = obs["diffusion_collect"]
            _idx = env_id if dc["body_pos"].ndim > 1 else slice(None)
            body_pos = np.asarray(dc["body_pos"][_idx].cpu(), dtype=np.float32).reshape(30, 3)
            body_quat = np.asarray(dc["body_ori"][_idx].cpu(), dtype=np.float32).reshape(30, 4)
            body_lin_vel = np.asarray(dc["body_lin_vel"][_idx].cpu(), dtype=np.float32).reshape(30, 3)
            body_ang_vel = np.asarray(dc["body_ang_vel"][_idx].cpu(), dtype=np.float32).reshape(30, 3)
            joint_pos = np.asarray(dc["dof_pos"][_idx].cpu(), dtype=np.float32)
            joint_vel = np.asarray(dc["dof_vel"][_idx].cpu(), dtype=np.float32)

            # agent stepping (diffusion policy)
            last_action_isaac = policy.get_action(
                body_pos, body_quat, body_lin_vel, body_ang_vel, joint_pos, joint_vel
            )

            # Convert to env action format
            action_env = _isaac_action_to_env(last_action_isaac, env, env_id)

            # Broadcast to all envs if needed
            if action_env.shape[0] < env.unwrapped.num_envs:
                action_env = action_env.repeat(env.unwrapped.num_envs, 1)

            # env stepping
            obs, _, _, _ = env.step(action_env)

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

    # close the simulator
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
