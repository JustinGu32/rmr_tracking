"""Script to collect rollouts into a zarr dataset compatible with DiffuseCloC."""
""" 
ENABLE_CAMERAS=1 python scripts/collect_dataset.py \
  --task=Tracking-Flat-G1-v0 \
  --wandb_path=justingu-stanford-university/stair_climbing/rmj7wc1b \
  --num_eps_collect=500 \
  --num_steps_collect=200 \
  --noise_level=0.5 \
  --save_folder=./datasets \
  --headless
"""
"""Launch Isaac Sim Simulator first."""

import argparse
import os
import pathlib
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
from isaaclab.app import AppLauncher

# Set camera environment variable before any Isaac Lab imports
os.environ["ENABLE_CAMERAS"] = "1"

# local imports
# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), "rsl_rl"))
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Collect dataset with RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during collection.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")

parser.add_argument("--checkpoint_file", type=str, default=None, help="Path to checkpoint file.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--save_folder", type=str, default="./datasets", help="Folder to save dataset.")
parser.add_argument("--num_eps_collect", type=int, default=500, help="Number of episodes to collect.")
parser.add_argument("--num_steps_collect", type=int, default=200, help="Number of steps per episode.")
parser.add_argument("--episode_collect_length", type=float, default=None, help="Episode length in seconds.")
parser.add_argument("--min_sample_idx", type=int, default=None, help="Minimum motion sample index.")
parser.add_argument("--max_sample_idx", type=int, default=None, help="Maximum motion sample index.")
parser.add_argument("--noise_level", type=float, default=0.5, help="Action noise level.")
parser.add_argument("--hip_noise", type=float, default=0.5, help="Hip joint noise level.")
parser.add_argument("--knee_noise", type=float, default=0.5, help="Knee joint noise level.")
parser.add_argument("--ankle_noise", type=float, default=0.7, help="Ankle joint noise level.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
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

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner

DIFFUSION_POLICY_ROOT = Path("/move/u/justingu/Projects/TML-BeyondMimic")
if DIFFUSION_POLICY_ROOT.exists():
    sys.path.insert(0, str(DIFFUSION_POLICY_ROOT))

from diffusion_policy.utils.replay_buffer import ReplayBuffer  # noqa: E402


@dataclass
class CollectConfig:
    """Command-line configuration for dataset collection."""

    task: str
    wandb_run_path: Optional[str] = None
    checkpoint_file: Optional[str] = None
    motion_file: Optional[str] = None
    save_folder: str = "./datasets"
    num_envs: Optional[int] = None
    device: Optional[str] = None
    video: bool = False
    video_length: int = 200
    num_eps_collect: int = 500
    num_steps_collect: int = 200
    episode_collect_length: Optional[float] = None
    min_sample_idx: Optional[int] = None
    max_sample_idx: Optional[int] = None
    noise_level: float = 0.5
    hip_noise: float = 0.5
    knee_noise: float = 0.5
    ankle_noise: float = 0.7


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Collect dataset with RSL-RL agent."""
    cfg = CollectConfig(
        task=args_cli.task,
        wandb_run_path=args_cli.wandb_path,
        checkpoint_file=args_cli.checkpoint_file,
        motion_file=args_cli.motion_file,
        save_folder=args_cli.save_folder,
        num_envs=args_cli.num_envs,
        device=args_cli.device,
        video=args_cli.video,
        video_length=args_cli.video_length,
        num_eps_collect=args_cli.num_eps_collect,
        num_steps_collect=args_cli.num_steps_collect,
        episode_collect_length=args_cli.episode_collect_length,
        min_sample_idx=args_cli.min_sample_idx,
        max_sample_idx=args_cli.max_sample_idx,
        noise_level=args_cli.noise_level,
        hip_noise=args_cli.hip_noise,
        knee_noise=args_cli.knee_noise,
        ankle_noise=args_cli.ankle_noise,
    )

    run_collection(cfg, env_cfg, agent_cfg)


def run_collection(cfg: CollectConfig, env_cfg, agent_cfg):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Override configs with CLI arguments
    env_cfg.scene.num_envs = cfg.num_envs if cfg.num_envs is not None else env_cfg.scene.num_envs
    if cfg.episode_collect_length is not None:
        env_cfg.episode_length_s = float(cfg.episode_collect_length)

    if cfg.min_sample_idx is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.min_sample_idx = cfg.min_sample_idx
    if cfg.max_sample_idx is not None and hasattr(env_cfg.commands, "motion"):
        env_cfg.commands.motion.max_sample_idx = cfg.max_sample_idx

    # Load motion file from wandb or CLI
    if cfg.wandb_run_path:
        configure_motion_file_wandb(env_cfg, cfg.wandb_run_path)
    elif cfg.motion_file:
        env_cfg.commands.motion.motion_file = cfg.motion_file

    render_mode = "rgb_array" if cfg.video else None
    
    import gymnasium as gym
    env = gym.make(cfg.task, cfg=env_cfg, device=device, render_mode=render_mode)

    # wrap for video recording if needed
    if cfg.video:
        video_kwargs = {
            "video_folder": os.path.join("videos", "collect"),
            "step_trigger": lambda step: step == 0,
            "video_length": cfg.video_length,
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env)

    # Load checkpoint
    resume_path, run_name = resolve_checkpoint(cfg, agent_cfg)

    # Create runner and load policy
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(str(resume_path))
    policy = runner.get_inference_policy(device=device)

    # Get observation and action dimensions
    obs, extra = env.reset()
    num_envs = env.unwrapped.num_envs
    action_dim = env.unwrapped.action_space.shape[-1]

    # Use critic observation group (privileged observations for dataset)
    obs_key = "critic"
    # Access underlying observation buffer since wrapper returns tensor
    full_obs = env.unwrapped.obs_buf
    if obs_key not in full_obs:
        available_obs_keys = list(full_obs.keys())
        raise KeyError(
            f"Expected observation group '{obs_key}' in observations, "
            f"but it was not found. Available observation groups: {available_obs_keys}"
        )

    # Get dimensions from privileged observations
    critic_obs = full_obs[obs_key]
    obs_dim = critic_obs.shape[-1]
    max_len = int(cfg.num_steps_collect)
    target_episodes = int(cfg.num_eps_collect)

    # Initialize recording buffers
    recorded_obs_episode = np.zeros((num_envs, max_len, obs_dim), dtype=np.float32)
    recorded_acs_episode = np.zeros((num_envs, max_len, action_dim), dtype=np.float32)

    # Camera dimensions from env config (D435: 848x480)
    camera_height = 480
    camera_width = 848
    recorded_rgb_episode = np.zeros((num_envs, max_len, camera_height, camera_width, 3), dtype=np.uint8)
    recorded_depth_episode = np.zeros((num_envs, max_len, camera_height, camera_width, 1), dtype=np.float32)
    episode_lengths = np.zeros(num_envs, dtype=np.int32)

    collected_obs: list[np.ndarray] = []
    collected_acs: list[np.ndarray] = []
    collected_rgb: list[np.ndarray] = []
    collected_depth: list[np.ndarray] = []

    # Track whether camera data is being collected (mutable container for nested function access)
    camera_available = [True]

    # Noise state for action perturbation
    noise_state = torch.zeros((num_envs, action_dim), device=device)
    theta = 0.0
    mu = 0.0
    dt = 1.0
    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    # Get joint index groups for noise
    hip_idxs, knee_idxs, ankle_idxs = joint_index_groups(env)

    try:
        while len(collected_obs) < target_episodes:
            with torch.no_grad():
                actions = policy(obs)
                clean_actions = actions.clone()

                # Add noise to actions
                sigma_noise = torch.randn_like(actions) * cfg.noise_level
                if hip_idxs:
                    sigma_noise[:, hip_idxs] = (
                        torch.randn((num_envs, len(hip_idxs)), device=device) * cfg.hip_noise
                    )
                if knee_idxs:
                    sigma_noise[:, knee_idxs] = (
                        torch.randn((num_envs, len(knee_idxs)), device=device) * cfg.knee_noise
                    )
                if ankle_idxs:
                    sigma_noise[:, ankle_idxs] = (
                        torch.randn((num_envs, len(ankle_idxs)), device=device) * cfg.ankle_noise
                    )

                # Update noise state (Ornstein-Uhlenbeck process)
                noise_state = noise_state + theta * (mu - noise_state) * dt
                noise_state = noise_state + sigma_noise * sqrt_dt
                actions = actions + noise_state

                # Record step data
                record_step(
                    env.unwrapped.obs_buf,
                    clean_actions,
                    recorded_obs_episode,
                    recorded_acs_episode,
                    recorded_rgb_episode,
                    recorded_depth_episode,
                    episode_lengths,
                    max_len,
                    obs_key,
                    env,
                    camera_available,
                )

                # Step environment
                obs, reward, dones, extra = env.step(actions)

                # Process completed episodes
                process_dones(
                    dones,
                    env.unwrapped.obs_buf,
                    extra,
                    recorded_obs_episode,
                    recorded_acs_episode,
                    recorded_rgb_episode,
                    recorded_depth_episode,
                    episode_lengths,
                    collected_obs,
                    collected_acs,
                    collected_rgb,
                    collected_depth,
                    max_len,
                    target_episodes,
                    obs_key,
                    camera_available,
                )

    finally:
        env.close()

    if not collected_obs:
        raise RuntimeError("No episodes were collected. Please check the rollout logs.")

    # Save dataset
    output_path = build_output_path(cfg, run_name, max_len, target_episodes)
    write_dataset(output_path, collected_obs, collected_acs, collected_rgb, collected_depth)
    print(f"[INFO] Saved dataset to {output_path}")


def resolve_checkpoint(cfg: CollectConfig, agent_cfg: RslRlOnPolicyRunnerCfg) -> tuple[Path, str]:
    """Resolve checkpoint path and run name."""
    log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    log_root_path.mkdir(parents=True, exist_ok=True)

    if cfg.wandb_run_path:
        resume_path, run_name = download_wandb_checkpoint(cfg.wandb_run_path, log_root_path)
        return resume_path, run_name

    if cfg.checkpoint_file:
        resume_path = Path(cfg.checkpoint_file).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {resume_path}")
        return resume_path, resume_path.stem

    # Use default checkpoint path
    resume_path = get_checkpoint_path(
        log_path=str(log_root_path),
        run_dir=agent_cfg.load_run,
        checkpoint=agent_cfg.load_checkpoint,
    )
    run_name = resume_path.parent.name
    return resume_path, run_name


def download_wandb_checkpoint(run_path: str, log_root_path: Path):
    """Download checkpoint from wandb."""
    import wandb

    api = wandb.Api()
    if "model" in run_path:
        run_path = "/".join(run_path.split("/")[:-1])
    wandb_run = api.run(run_path)

    model_files = [file.name for file in wandb_run.files() if file.name.startswith("model_")]
    if not model_files:
        raise RuntimeError(f"No model_*.pt files found in wandb run {run_path}")

    def extract_step(filename: str) -> int:
        step_part = filename.split("_")[1]
        return int(step_part.split(".")[0])

    latest_file = max(model_files, key=extract_step)
    download_dir = log_root_path / "wandb_checkpoints" / wandb_run.id
    download_dir.mkdir(parents=True, exist_ok=True)
    wandb_run.file(latest_file).download(str(download_dir), replace=True)
    resume_path = download_dir / latest_file
    return resume_path, wandb_run.name


def configure_motion_file_wandb(env_cfg, wandb_path: str) -> None:
    """Configure motion file from wandb artifact."""
    if not hasattr(env_cfg, "commands") or not hasattr(env_cfg.commands, "motion"):
        return

    import wandb

    api = wandb.Api()
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])
    else:
        run_path = wandb_path
    wandb_run = api.run(run_path)

    motion_artifact = next(
        (a for a in wandb_run.used_artifacts() if a.type in {"motions", "combined_motions"}),
        None,
    )
    if motion_artifact is None:
        print("[WARN] No motion artifact attached to this wandb run.")
        return

    artifact_dir = Path(motion_artifact.download())
    motion_file_name = Path(motion_artifact.file()).name
    motion_path = artifact_dir / motion_file_name

    if not motion_path.exists():
        npz_files = list(artifact_dir.glob("*.npz"))
        if not npz_files:
            print("[WARN] Unable to locate motion file inside artifact download.")
            return
        motion_path = npz_files[0]

    env_cfg.commands.motion.motion_file = str(motion_path)


def joint_index_groups(env):
    """Get joint index groups for noise application."""
    joint_names = env.unwrapped.command_manager.get_term("motion").robot.joint_names
    hip_idxs = [idx for idx, name in enumerate(joint_names) if "hip" in name]
    knee_idxs = [idx for idx, name in enumerate(joint_names) if "knee_joint" in name]
    ankle_idxs = [idx for idx, name in enumerate(joint_names) if "ankle" in name]
    return hip_idxs, knee_idxs, ankle_idxs


def record_step(
    obs: dict,
    clean_actions: torch.Tensor,
    recorded_obs_episode: np.ndarray,
    recorded_acs_episode: np.ndarray,
    recorded_rgb_episode: np.ndarray,
    recorded_depth_episode: np.ndarray,
    episode_lengths: np.ndarray,
    max_len: int,
    obs_key: str,
    env,
    camera_available: list,
) -> None:
    """Record data for current step."""
    obs_tensor = obs[obs_key].detach().cpu().numpy()
    actions_np = clean_actions.detach().cpu().numpy()
    env_ids = np.arange(obs_tensor.shape[0])
    valid_mask = episode_lengths < max_len
    if not np.any(valid_mask):
        return
    valid_ids = env_ids[valid_mask]
    rows = episode_lengths[valid_mask]

    # Record obs and actions
    recorded_obs_episode[valid_ids, rows] = obs_tensor[valid_ids]
    recorded_acs_episode[valid_ids, rows] = actions_np[valid_ids]

    # Record camera data
    if camera_available[0]:
        try:
            camera_data = env.unwrapped.scene["depth_camera"].data
            rgb_data = camera_data.output["rgb"].detach().cpu().numpy()  # (num_envs, H, W, 3)
            depth_data = camera_data.output["depth"].detach().cpu().numpy()  # (num_envs, H, W, 1)

            recorded_rgb_episode[valid_ids, rows] = rgb_data[valid_ids]
            recorded_depth_episode[valid_ids, rows] = depth_data[valid_ids]
        except KeyError:
            print("[WARN] Camera data not available, disabling camera recording for this run")
            camera_available[0] = False
        except Exception as e:
            print(f"[WARN] Error accessing camera data: {e}, disabling camera recording for this run")
            camera_available[0] = False

    episode_lengths[valid_mask] += 1


def process_dones(
    dones: torch.Tensor,
    obs: dict,
    extra: dict,
    recorded_obs_episode: np.ndarray,
    recorded_acs_episode: np.ndarray,
    recorded_rgb_episode: np.ndarray,
    recorded_depth_episode: np.ndarray,
    episode_lengths: np.ndarray,
    collected_obs: list[np.ndarray],
    collected_acs: list[np.ndarray],
    collected_rgb: list[np.ndarray],
    collected_depth: list[np.ndarray],
    max_len: int,
    target_episodes: int,
    obs_key: str,
    camera_available: list,
) -> None:
    """Process completed episodes."""
    done_indices = torch.nonzero(dones, as_tuple=False).flatten().tolist()
    if not done_indices:
        return

    time_outs = extra.get("time_outs")
    if time_outs is None:
        time_outs = torch.ones_like(dones)

    for env_id in done_indices:
        current_len = min(int(episode_lengths[env_id]), max_len)
        successful = bool(time_outs[env_id])

        if successful and current_len >= max_len:
            # Save episode data
            obs_slice = recorded_obs_episode[env_id, :current_len].copy()
            act_slice = recorded_acs_episode[env_id, :current_len].copy()

            collected_obs.append(obs_slice)
            collected_acs.append(act_slice)

            if camera_available[0]:
                rgb_slice = recorded_rgb_episode[env_id, :current_len].copy()
                depth_slice = recorded_depth_episode[env_id, :current_len].copy()
                collected_rgb.append(rgb_slice)
                collected_depth.append(depth_slice)

            print(
                f"[INFO] Saved episode {len(collected_obs)}/{target_episodes} "
                f"(env {env_id}, length {current_len})"
            )

        # Reset episode buffers
        recorded_obs_episode[env_id] = 0.0
        recorded_acs_episode[env_id] = 0.0
        if camera_available[0]:
            recorded_rgb_episode[env_id] = 0.0
            recorded_depth_episode[env_id] = 0.0
        episode_lengths[env_id] = 0

        if len(collected_obs) >= target_episodes:
            break


def build_output_path(cfg: CollectConfig, run_name: str, steps: int, episodes: int) -> Path:
    """Build output path for dataset."""
    timestamp = datetime.now().strftime("%m%d_%H%M")
    base = (
        f"{run_name}_ep-{episodes}_steps-{steps}"
        f"_noise-{cfg.noise_level:.2f}_hip-{cfg.hip_noise:.2f}"
        f"_knee-{cfg.knee_noise:.2f}_ankle-{cfg.ankle_noise:.2f}_{timestamp}.zarr"
    )
    save_dir = Path(cfg.save_folder).expanduser().resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir / base


def write_dataset(
    output_path: Path,
    obs_episodes: list[np.ndarray],
    act_episodes: list[np.ndarray],
    rgb_episodes: list[np.ndarray],
    depth_episodes: list[np.ndarray],
) -> None:
    """Write dataset to zarr format compatible with DiffuseCloC."""
    buff = ReplayBuffer.create_empty_zarr()

    # For now, save the raw privileged observations and camera data
    # You may need to adapt this to extract individual components based on your specific needs
    for i, (obs_ep, act_ep) in enumerate(zip(obs_episodes, act_episodes)):
        episode_dict = {
            "obs": obs_ep,  # Raw privileged observations (critic group)
            "act": act_ep,
        }

        # Add camera data if available
        if i < len(rgb_episodes) and i < len(depth_episodes):
            episode_dict["rgb"] = rgb_episodes[i]   # Camera RGB images (T, H, W, 3)
            episode_dict["depth"] = depth_episodes[i]  # Camera depth images (T, H, W, 1)

        buff.add_episode(episode_dict)

    buff.save_to_path(str(output_path))


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()