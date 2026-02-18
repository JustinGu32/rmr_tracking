"""
Minimal sim2sim: diffusion policy in Isaac Lab.
Loads policy via wandb, gets obs, stacks history, calls act(), steps env.
No guidance, no ensemble, no inpainting -- bare minimum for debugging.

Usage:
  python scripts/sim2sim_isaaclab.py --task=Tracking-Flat-G1-v0 --wandb_path user/project/run_id
  python scripts/sim2sim_isaaclab.py --task=Tracking-Flat-G1-v0 --checkpoint path/to/ckpt.pt
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Isaac Lab boilerplate: AppLauncher must be created before other Isaac imports
# ---------------------------------------------------------------------------
if os.environ.get("ENABLE_CAMERAS", "") != "1":
    os.environ["ENABLE_CAMERAS"] = "0"

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
RMR_TRACKING_ROOT = SCRIPT_DIR.parent
TML_ROOT = RMR_TRACKING_ROOT.parent / "TML-BeyondMimic"
sys.path.insert(0, str(RMR_TRACKING_ROOT))
sys.path.insert(0, str(RMR_TRACKING_ROOT / "scripts" / "rsl_rl"))
sys.path.insert(0, str(TML_ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal diffusion policy sim2sim in Isaac Lab")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--wandb_path", type=str, default=None)
parser.add_argument("--wandb_file", type=str, default="latest.ckpt")
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--motion_file", type=str, default='/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz')
parser.add_argument("--video", action="store_true", help="Record simulation to a video file (offscreen; use with --headless on servers)")
parser.add_argument("--video_folder", type=str, default="videos/no_vision", help="Folder to save video (default: videos/no_vision)")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record (default: 500)")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if getattr(args_cli, "video", False):
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac Lab app ready.", flush=True)

# ---------------------------------------------------------------------------
# Imports after AppLauncher (Isaac Lab requirement)
# ---------------------------------------------------------------------------
import gymnasium as gym
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab_tasks.utils.hydra import hydra_task_config
import whole_body_tracking.tasks  # register tasks

from diffusion_policy.inference.diffusion_agent import load_wandb, load_diffusion_policy


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    # --- Validate args ---
    if args_cli.checkpoint is None and args_cli.wandb_path is None:
        print("[ERROR] Specify --checkpoint or --wandb_path"); sys.exit(1)
    if args_cli.checkpoint and args_cli.wandb_path:
        print("[ERROR] Specify only one of --checkpoint or --wandb_path"); sys.exit(1)

    device = args_cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 1. Load diffusion policy ---
    print("[INFO] Loading diffusion policy...", flush=True)
    if args_cli.wandb_path:
        model, cfg, _, run_name = load_wandb(args_cli.wandb_path, args_cli.wandb_file, device=device)
        print(f"[INFO] Loaded wandb run: {run_name}")
    else:
        model, cfg = load_diffusion_policy(args_cli.checkpoint, device=device)
    model.eval()

    if args_cli.deterministic:
        model.set_deterministic(True)

    # Reset rolling trajectory state
    model.action_rolling_traj = None
    model.state_rolling_traj = None

    n_past_steps = model.n_past_steps
    nom_frame_idx = n_past_steps - 1
    print(f"[INFO] Policy loaded. n_past_steps={n_past_steps}, action_dim={model.action_dim}, obs_dim={model.obs_dim}")

    # --- 2. Create Isaac Lab environment ---
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = args_cli.motion_file
        
    record_video = getattr(args_cli, "video", False)
    video_length = getattr(args_cli, "video_length", 500)
        
    # Relax termination thresholds: the diffusion policy generates
    # its own motion and does NOT track the reference motion file, so the reference-based
    # terminations (anchor_pos, anchor_ori, ee_body_pos) trigger spurious resets that
    # corrupt the policy's temporal observation buffer.
    steps_to_seconds = env_cfg.decimation * env_cfg.sim.dt
    episode_s = (max(args_cli.steps, video_length) + 200) * steps_to_seconds
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, episode_s)
    if hasattr(env_cfg.terminations, "anchor_pos") and hasattr(env_cfg.terminations.anchor_pos, "params"):
        env_cfg.terminations.anchor_pos.params["threshold"] = 10.0
    if hasattr(env_cfg.terminations, "anchor_ori") and hasattr(env_cfg.terminations.anchor_ori, "params"):
        env_cfg.terminations.anchor_ori.params["threshold"] = 10.0
    if hasattr(env_cfg.terminations, "ee_body_pos") and hasattr(env_cfg.terminations.ee_body_pos, "params"):
        env_cfg.terminations.ee_body_pos.params["threshold"] = 10.0
    print(f"[INFO] Relaxed termination thresholds for sim2sim (episode_length_s={env_cfg.episode_length_s:.1f})", flush=True)
    
    render_mode = "rgb_array" if record_video else None
    print(f"[INFO] Creating environment (render_mode={render_mode!r}, may take 1-2 min)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    print("[INFO] Environment created.", flush=True)
    
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(getattr(args_cli, "video_folder", "videos/takara_sim2sim_isaaclab")))
        video_length = getattr(args_cli, "video_length", 500)
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )
        print(f"[INFO] Video recording: first {video_length} steps -> {video_folder}")

    # --- 3. Observation history buffer ---
    obs_buffer = deque(maxlen=n_past_steps)

    def extract_obs(obs_raw, env_id=0):
        """
        Extract raw body/joint state from Isaac Lab obs dict.
        diffusion_collect has concatenate_terms=False so it's already a dict.
        Keys: body_pos (N,90), body_ori (N,120), body_lin_vel (N,90),
              body_ang_vel (N,90), dof_pos (N,29), dof_vel (N,29)
        """
        dc = obs_raw['diffusion_collect']
        return {
            'body_pos':     dc['body_pos'][env_id].float().reshape(30, 3),
            'body_rot':     dc['body_ori'][env_id].float().reshape(30, 4),
            'body_lin_vel': dc['body_lin_vel'][env_id].float().reshape(30, 3),
            'body_ang_vel': dc['body_ang_vel'][env_id].float().reshape(30, 3),
            'joint_pos':    dc['dof_pos'][env_id].float(),    # (29,)
            'joint_vel':    dc['dof_vel'][env_id].float(),    # (29,)
        }

    def build_obs_dict():
        """
        Stack obs_buffer into batched obs_dict for act().
        Returns dict with each value shaped (1, H, ...).
        """
        obs_dict = {}
        for key in obs_buffer[0].keys():
            # Stack along time dim -> (H, ...), add batch dim -> (1, H, ...)
            obs_dict[key] = torch.stack([o[key] for o in obs_buffer]).unsqueeze(0).to(device)
        return obs_dict

    # --- 4. Run loop ---
    print("[INFO] Resetting environment...", flush=True)
    obs_raw, _ = env.reset()
    print("[INFO] Environment reset. Starting control loop.", flush=True)

    max_steps = args_cli.steps
    step_count = 0

    while step_count < max_steps and simulation_app.is_running():
        # Extract single-frame observation (already reshaped)
        obs_frame = extract_obs(obs_raw)

        # Add to buffer; pad with first obs if buffer not full yet
        obs_buffer.append(obs_frame)
        while len(obs_buffer) < n_past_steps:
            obs_buffer.appendleft(obs_buffer[0])

        # Build batched obs_dict: {key: (1, H, ...)}
        obs_dict = build_obs_dict()

        # --- Call act() ---
        # this does NOT use the DiffusionAgent
        with torch.no_grad():
            action_traj, state_traj = model.act(
                obs_dict=obs_dict,
                nom_frame_idx=nom_frame_idx,
            )

        # Extract action at nominal frame and unnormalize
        action_normalized = action_traj[0, nom_frame_idx]  # (action_dim,)
        action = model.normalizer.unnormalize(
            {'action': action_normalized.unsqueeze(0)}
        )['action'].squeeze(0)  # (action_dim,)

        # --- Step environment ---
        action_env = action.unsqueeze(0)  # (1, action_dim)
        if action_env.shape[0] < env.unwrapped.num_envs:
            action_env = action_env.repeat(env.unwrapped.num_envs, 1)

        obs_raw, _, _, _, _ = env.step(action_env)
        step_count += 1

        # Print progress
        if step_count % 100 == 0:
            robot = env.unwrapped.scene["robot"]
            pelvis_z = robot.data.body_pos_w[0, 0, 2].item()
            print(f"Step {step_count}/{max_steps}: pelvis height = {pelvis_z:.3f}m")

    # --- Done ---
    robot = env.unwrapped.scene["robot"]
    pelvis_z = robot.data.body_pos_w[0, 0, 2].item()
    print(f"\n[INFO] Done. {step_count} steps. Final pelvis height = {pelvis_z:.3f}m")
    if pelvis_z < 0.3:
        print("[WARNING] Robot appears to have fallen")
    else:
        print("[SUCCESS] Robot maintained balance")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
