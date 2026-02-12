"""
Run diffusion policy in Isaac Lab (G1 Tracking-Flat) instead of MuJoCo.
Same logic as sim2sim.py: load policy, run control loop with optional guidance,
but simulation is Isaac Lab as in rmr_tracking/scripts/collect_dataset.py and play.py.

Usage (from rmr_tracking or with PYTHONPATH including rmr_tracking):
  python scripts/sim2sim_mine_isaaclab.py --task=Tracking-Flat-G1-v0 --checkpoint ckpts/diffusion_policy_latest.pt
  python scripts/sim2sim_mine_isaaclab.py --task=Tracking-Flat-G1-v0 --wandb_path user/project/run_id

Headless + save video (for servers without display):
  --headless          Disable interactive viewer (required on servers).
  --video             Record simulation to a video file (uses offscreen rendering).
  --video_folder DIR  Where to save the video (default: videos/sim2sim_isaaclab).
  --video_length N    Steps to record (default: 500).
  Example: python scripts/sim2sim_mine_isaaclab.py --task=Tracking-Flat-G1-v0 --checkpoint ckpts/foo.pt --headless --video --video_folder ./out
"""

import argparse
import os
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import torch

# Optional: set cameras before any Isaac Lab imports
if os.environ.get("ENABLE_CAMERAS", "") != "1":
    os.environ["ENABLE_CAMERAS"] = "0"

# Add rmr_tracking for task registration (collect_dataset / play style)
TML_ROOT = Path(__file__).resolve().parent.parent
RMR_TRACKING_ROOT = TML_ROOT.parent / "rmr_tracking"
if RMR_TRACKING_ROOT.exists():
    sys.path.insert(0, str(RMR_TRACKING_ROOT))
    sys.path.insert(0, str(RMR_TRACKING_ROOT / "scripts" / "rsl_rl"))
else:
    RMR_TRACKING_ROOT = Path(os.environ.get("RMR_TRACKING_ROOT", ""))
    if RMR_TRACKING_ROOT.exists():
        sys.path.insert(0, str(RMR_TRACKING_ROOT))
        sys.path.insert(0, str(RMR_TRACKING_ROOT / "scripts" / "rsl_rl"))

# Isaac Lab app must be launched before other Isaac imports (play.py / collect_dataset.py pattern)
from isaaclab.app import AppLauncher

# CLI: diffusion args + task/num_envs + AppLauncher
parser = argparse.ArgumentParser(description="Run diffusion policy in Isaac Lab (G1)")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint (.pt)")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path (e.g. user/project/run_id)")
parser.add_argument("--wandb_file", type=str, default="latest.ckpt", help="Checkpoint filename in wandb")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
# parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic sampling")
parser.add_argument("--guidance_type", type=str, default=None, help="Guidance type (e.g. joystick, target_heading)")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Isaac Lab task (e.g. Tracking-Flat-G1-v0)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str, default="/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz", help="Path to motion file for tracking command")
parser.add_argument("--video", action="store_true", help="Record simulation to a video file (offscreen; use with --headless on servers)")
parser.add_argument("--video_folder", type=str, default="videos/no_vision", help="Folder to save video (default: videos/no_vision)")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record (default: 500)")
# Adds --headless, --device_id, etc. (use --headless on servers without a display)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Enable cameras for offscreen rendering when recording video (required for headless + video)
if getattr(args_cli, "video", False):
    args_cli.enable_cameras = True

# Clear sys.argv for Hydra (collect_dataset / play pattern)
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse app first
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac Lab app ready.", flush=True)

# Rest of imports after app launch (play.py / collect_dataset.py pattern)
import gymnasium as gym

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# Register tasks (G1 Tracking-Flat, etc.)
import whole_body_tracking.tasks  # noqa: E402, F401

# Diffusion policy (TML-BeyondMimic)
sys.path.insert(0, str(TML_ROOT))
from diffusion_policy.inference.guidance import create_guidance_fn  # noqa: E402
from diffusion_policy.inference.diffusion_agent import DiffusionAgentIsaac  # noqa: E402

# Keyboard joystick for guidance (same as sim2sim.py)
# try:
#     from pynput import keyboard
#     PYNPUT_AVAILABLE = True
# except ImportError:
#     PYNPUT_AVAILABLE = False
#     print("[WARNING] pynput not available - keyboard joystick disabled (pip install pynput)")

# Isaac joint scale and default pose (policy outputs 29 dims; target = action * scale + default)
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

# Isaac body order (from map_npz_mj2isaac) to index robot.data by name
ISAAC_BODY_NAMES = [
    "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
    "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link", "left_hip_yaw_link",
    "right_hip_yaw_link", "torso_link", "left_knee_link", "right_knee_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link", "left_ankle_pitch_link",
    "right_ankle_pitch_link", "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_ankle_roll_link", "right_ankle_roll_link", "left_shoulder_yaw_link",
    "right_shoulder_yaw_link", "left_elbow_link", "right_elbow_link",
    "left_wrist_roll_link", "right_wrist_roll_link", "left_wrist_pitch_link",
    "right_wrist_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link",
]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class KeyboardJoystick:
    """Keyboard joystick for guidance (same as sim2sim.py)."""
    def __init__(self):
        self.lx = self.ly = self.rx = self.ry = 0.0
        self._keys_pressed = set()
        self._lock = Lock()
        if PYNPUT_AVAILABLE:
            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.start()
            print("[KEYBOARD] Joystick: Up/Down=forward/back, Left/Right=strafe, </>=rotate")
        else:
            print("[KEYBOARD] Disabled (pynput not installed)")

    def _on_press(self, key):
        with self._lock:
            if hasattr(key, "char"):
                self._keys_pressed.add(key.char)
            else:
                self._keys_pressed.add(key)
            self._update_joystick()

    def _on_release(self, key):
        with self._lock:
            if hasattr(key, "char"):
                self._keys_pressed.discard(key.char)
            else:
                self._keys_pressed.discard(key)
            self._update_joystick()

    def _update_joystick(self):
        self.lx = self.ly = self.rx = self.ry = 0.0
        if keyboard.Key.up in self._keys_pressed:
            self.ly = 1.0
        if keyboard.Key.down in self._keys_pressed:
            self.ly = -1.0
        if keyboard.Key.left in self._keys_pressed:
            self.lx = -1.0
        if keyboard.Key.right in self._keys_pressed:
            self.lx = 1.0
        if "," in self._keys_pressed or "<" in self._keys_pressed:
            self.rx = 1.0
        if "." in self._keys_pressed or ">" in self._keys_pressed:
            self.rx = -1.0

    def stop(self):
        if PYNPUT_AVAILABLE and hasattr(self, "_listener"):
            self._listener.stop()


def _robot_state_isaac(env, env_id=0):
    """Read robot state from Isaac env in Isaac order for DiffusionAgentIsaac (no MuJoCo conversion)."""
    robot = env.unwrapped.scene["robot"]
    body_names = list(robot.body_names)
    isaac_body_indices = []
    for name in ISAAC_BODY_NAMES:
        if name in body_names:
            isaac_body_indices.append(body_names.index(name))
        else:
            raise KeyError(f"Robot missing body {name!r}. Have: {body_names[:5]}...")
    isaac_body_indices = np.array(isaac_body_indices)

    def get_bodies(x):
        xi = x[env_id].cpu().numpy()
        return xi[isaac_body_indices].astype(np.float32)

    body_pos = get_bodies(robot.data.body_pos_w)
    body_quat = get_bodies(robot.data.body_quat_w)
    body_lin_vel = get_bodies(robot.data.body_lin_vel_w)
    body_ang_vel = get_bodies(robot.data.body_ang_vel_w)
    joint_pos = robot.data.joint_pos[env_id].cpu().numpy().astype(np.float32)
    joint_vel = robot.data.joint_vel[env_id].cpu().numpy().astype(np.float32)
    return body_pos, body_quat, body_lin_vel, body_ang_vel, joint_pos, joint_vel


def _isaac_action_to_env(action_isaac, env, env_id=0):
    """Convert policy output (Isaac order, unnormalized target positions) to env action (offset + scale)."""
    robot = env.unwrapped.scene["robot"]
    # Policy returns unnormalized action in Isaac order; target = action * scale + default
    target_isaac = action_isaac * ACTION_SCALE_ISAAC + DEFAULT_POSE_ISAAC
    target_isaac = torch.tensor(target_isaac, device=robot.device, dtype=torch.float32)
    # Env expects: joint target = offset + scale * action_env, so action_env = (target - offset) / scale
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    scale = action_term._scale[env_id]
    offset = action_term._offset[env_id]
    action_env = (target_isaac - offset) / scale
    return action_env.unsqueeze(0)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Run diffusion policy in Isaac Lab (same flow as sim2sim.py, sim from collect_dataset/play)."""
    print("[INFO] main() entered (Hydra config loaded).", flush=True)
    # Validate checkpoint args
    if args_cli.checkpoint is None and args_cli.wandb_path is None:
        print(f"[ERROR] No checkpoint specified")
        sys.exit(1)
    if args_cli.checkpoint and args_cli.wandb_path:
        print("[ERROR] Specify only one of --checkpoint or --wandb_path")
        sys.exit(1)

    device = args_cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Create env (play.py / collect_dataset style); use rgb_array for video (works headless)
    record_video = getattr(args_cli, "video", False)
    video_length = getattr(args_cli, "video_length", 500)
    # When recording, use one long episode so the video is one continuous trajectory (no resets)
    if record_video:
        steps_to_seconds = env_cfg.decimation * env_cfg.sim.dt
        episode_s = (video_length + 200) * steps_to_seconds  # buffer so we don't hit time_out
        env_cfg.episode_length_s = max(env_cfg.episode_length_s, episode_s)
        # Relax failure terminations so policy drift doesn't cause resets mid-video
        if hasattr(env_cfg.terminations, "anchor_pos") and hasattr(env_cfg.terminations.anchor_pos, "params"):
            env_cfg.terminations.anchor_pos.params["threshold"] = 10.0
        if hasattr(env_cfg.terminations, "anchor_ori") and hasattr(env_cfg.terminations.anchor_ori, "params"):
            env_cfg.terminations.anchor_ori.params["threshold"] = 10.0
        if hasattr(env_cfg.terminations, "ee_body_pos") and hasattr(env_cfg.terminations.ee_body_pos, "params"):
            env_cfg.terminations.ee_body_pos.params["threshold"] = 10.0
        print(f"[INFO] Video mode: episode_length_s={env_cfg.episode_length_s:.1f} for continuous recording", flush=True)

    render_mode = "rgb_array" if record_video else None
    print(f"[INFO] Creating environment (render_mode={render_mode!r}, may take 1-2 min)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)
    print("[INFO] Environment created.", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(getattr(args_cli, "video_folder", "videos/sim2sim_isaaclab")))
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

    # Load diffusion policy (Isaac ordering: no MuJoCo conversion inside agent)
    print("[INFO] Loading diffusion policy (DiffusionAgentIsaac); wandb download can be slow...", flush=True)
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
    print("[INFO] Diffusion policy loaded (Isaac ordering).", flush=True)
    
    # Guidance
    guidance_fn = None
    keyboard_joystick = None
    if args_cli.guidance_type and args_cli.guidance_scale > 0.0:
        guidance_config = {"target_velocity": [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
        guidance_fn = create_guidance_fn(args_cli.guidance_type, guidance_config, torch.device(device))
        policy.actor.guidance_inpaint_nominal_state = False
        print(f"[GUIDANCE] {args_cli.guidance_type} scale={args_cli.guidance_scale}")
        if args_cli.guidance_type == "joystick":
            keyboard_joystick = KeyboardJoystick()

    policy.reset()
    print("[INFO] Resetting environment (first step may be slow)...", flush=True)
    obs, _ = env.reset()
    print("[INFO] Environment reset; starting control loop.", flush=True)
    step_count = 0
    max_steps = args_cli.steps
    env_id = 0
    last_action_isaac = None

    while step_count < max_steps and simulation_app.is_running():
        # Get state from diffusion_collect (flattened); reshape to (30, 3), (30, 4) for policy
        dc = obs['diffusion_collect']
        _idx = env_id if dc['body_pos'].ndim > 1 else slice(None)
        body_pos = np.asarray(dc['body_pos'][_idx].cpu(), dtype=np.float32).reshape(30, 3)
        body_quat = np.asarray(dc['body_ori'][_idx].cpu(), dtype=np.float32).reshape(30, 4)
        body_lin_vel = np.asarray(dc['body_lin_vel'][_idx].cpu(), dtype=np.float32).reshape(30, 3)
        body_ang_vel = np.asarray(dc['body_ang_vel'][_idx].cpu(), dtype=np.float32).reshape(30, 3)
        joint_pos = np.asarray(dc['dof_pos'][_idx].cpu(), dtype=np.float32)
        joint_vel = np.asarray(dc['dof_vel'][_idx].cpu(), dtype=np.float32)

        # Query policy at decimation rate
        if step_count % env.unwrapped.cfg.decimation == 0:
        
            if guidance_fn is not None:
                if args_cli.guidance_type == "joystick" and keyboard_joystick is not None:
                    if hasattr(guidance_fn, "joystick_values"):
                        guidance_fn.joystick_values[0] = keyboard_joystick.lx
                        guidance_fn.joystick_values[1] = keyboard_joystick.ly
                        guidance_fn.joystick_values[2] = keyboard_joystick.rx
                        guidance_fn.joystick_values[3] = keyboard_joystick.ry
                last_action_isaac = policy.get_action(
                    body_pos, body_quat, body_lin_vel, body_ang_vel,
                    joint_pos, joint_vel,
                    guidance_fn=guidance_fn,
                    guidance_kwargs=None,
                    guidance_scale=args_cli.guidance_scale,
                )
            else:
                last_action_isaac = policy.get_action(
                    body_pos, body_quat, body_lin_vel, body_ang_vel,
                    joint_pos, joint_vel,
                )
            # if last_action_isaac is None:
            #     last_action_isaac = np.zeros(29, dtype=np.float32)
        action_env = torch.tensor(last_action_isaac, device=device, dtype=torch.float32).unsqueeze(0)
        # action_env = _isaac_action_to_env(last_action_isaac, env, env_id)

        # Step env (vec: obs is batched)
        if action_env.shape[0] < env.unwrapped.num_envs:
            action_env = action_env.repeat(env.unwrapped.num_envs, 1)
        # breakpoint()
        obs, _, _, _, _ = env.step(action_env)
        step_count += 1

        if step_count % 500 == 0:
            robot = env.unwrapped.scene["robot"]
            pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
            print(f"Step {step_count}: pelvis height = {pelvis_z:.3f}m")

    # loop done
    if keyboard_joystick is not None:
        keyboard_joystick.stop()

    # Final stats before closing env
    print(f"[INFO] Completed {step_count} steps")
    robot = env.unwrapped.scene["robot"]
    pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
    print(f"[INFO] Final pelvis height = {pelvis_z:.3f}m")
    if pelvis_z < 0.3:
        print("[WARNING] Robot appears to have fallen")
    else:
        print("[SUCCESS] Robot maintained balance")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
