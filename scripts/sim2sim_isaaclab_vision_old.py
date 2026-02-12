"""
Run vision-conditioned diffusion policy in Isaac Lab (G1 Tracking-Flat).

Same as sim2sim_isaaclab but adds RGB + depth from an Isaac Lab camera, encoded with
SigLIP2 (RGB) and DeFM (depth), and passes vision_embeds to the policy. Requires
--enable_cameras (set automatically by this script).

Usage (from rmr_tracking or with PYTHONPATH including rmr_tracking and TML-BeyondMimic):
  python scripts/sim2sim_isaaclab_vision.py --task=Tracking-Flat-G1-v0 --checkpoint ckpts/vision_ckpt.pt
  python scripts/sim2sim_isaaclab_vision.py --task=Tracking-Flat-G1-v0 --wandb_path user/project/run_id

Headless + video:
  --headless --video   (cameras are enabled automatically for vision)

Debug robot vision (print stats + save RGB/depth images):
  --debug_vision       (saves to <video_folder>/vision_debug/ at steps 0, 100, 200, ...)
"""

import argparse
import os
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import torch

# Vision script requires cameras (RGB + depth). Set before any Isaac Lab imports so
# the app and any scene config that checks ENABLE_CAMERAS enable camera support.
os.environ["ENABLE_CAMERAS"] = "1"

# Add rmr_tracking for task registration (collect_dataset / play style)
TML_ROOT = Path(__file__).resolve().parent.parent
RMR_TRACKING_ROOT = TML_ROOT.parent / "rmr_tracking"

# Use unitree_ros G1 URDF (has d435_link) when not set in env. Path: repo_root/unitree_ros/...
if "G1_URDF_PATH" not in os.environ:
    _g1_urdf = TML_ROOT.parent / "unitree_ros" / "robots" / "g1_description" / "g1_29dof.urdf"
    if _g1_urdf.exists():
        os.environ["G1_URDF_PATH"] = str(_g1_urdf.resolve())
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
parser.add_argument("--steps", type=int, default=1000, help="Number of simulation steps")
# parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic sampling")
parser.add_argument("--guidance_type", type=str, default=None, help="Guidance type (e.g. joystick, target_heading)")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Isaac Lab task (e.g. Tracking-Flat-G1-v0)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str, default="/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz", help="Path to motion file for tracking command")
parser.add_argument("--video", action="store_true", help="Record simulation to a video file (offscreen; use with --headless on servers)")
parser.add_argument("--video_folder", type=str, default="videos/vision", help="Folder to save video (default: videos/vision)")
parser.add_argument("--video_length", type=int, default=1000, help="Number of steps to record (default: 1000)")
parser.add_argument("--debug_vision", action="store_true", help="Print and save robot vision (RGB/depth) for debugging")
# Adds --headless, --device_id, etc. (use --headless on servers without a display)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Vision script always needs cameras (RGB + depth for policy). Also enable for --video recording.
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

# Isaac Lab camera for vision (RGB + depth)
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors import TiledCameraCfg

# Diffusion policy (TML-BeyondMimic)
sys.path.insert(0, str(TML_ROOT))
from diffusion_policy.inference.guidance import create_guidance_fn  # noqa: E402
from diffusion_policy.inference.diffusion_agent import DiffusionAgentIsaac  # noqa: E402

# Keyboard joystick for guidance (same as sim2sim.py)
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    keyboard = None
    print("[WARNING] pynput not available - keyboard joystick disabled (pip install pynput)")

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

# Vision embedding dims: SigLIP2 1152 + DeFM 1024
RGB_EMBED_DIM = 1152
DEPTH_EMBED_DIM = 1024
VISION_EMBED_DIM = RGB_EMBED_DIM + DEPTH_EMBED_DIM  # 2176


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
    """Read robot state from Isaac env in Isaac order for DiffusionAgentIsaac."""
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
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    scale = action_term._scale[env_id]
    offset = action_term._offset[env_id]
    action_env = (target_isaac - offset) / scale
    return action_env.unsqueeze(0)


def load_vision_encoders(device: str = "cuda"):
    """Load SigLIP2 (RGB) and DeFM (depth) for vision embeddings."""
    siglip_model, siglip_processor = None, None
    defm_model = None

    try:
        from transformers import AutoImageProcessor, AutoModel
        ckpt = "google/siglip2-so400m-patch14-384"
        siglip_model = AutoModel.from_pretrained(ckpt, device_map=device).eval()
        # Use AutoImageProcessor: SigLIP2 is vision-only and AutoProcessor fails due to
        # transformers bug (TOKENIZER_MAPPING_NAMES returns None for Siglip2).
        siglip_processor = AutoImageProcessor.from_pretrained(ckpt)
        print(f"[VISION] Loaded SigLIP2 from {ckpt}")
    except Exception as e:
        print(f"[VISION] Failed to load SigLIP: {e}")
        raise

    try:
        torch.hub.set_dir("/move/u/chrzhang/.cache/torch/hub")
        defm_model = torch.hub.load(
            "leggedrobotics/defm:main", "defm_vit_l14", pretrained=True, trust_repo=True
        )
        defm_model = defm_model.eval().to(device)
        print("[VISION] Loaded DeFM depth encoder")
    except Exception as e:
        print(f"[VISION] Failed to load DeFM: {e}")
        raise

    return siglip_model, siglip_processor, defm_model


def encode_rgb(rgb: np.ndarray, model, processor, device: str) -> np.ndarray:
    """Encode single RGB image (H, W, 3) to (1152,) with SigLIP."""
    from PIL import Image
    
    # Ensure RGB is uint8 [0, 255] format
    if rgb.dtype == np.uint8:
        # MuJoCo returns uint8, but ensure it's in valid range
        rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        # Convert float to uint8
        if rgb.max() <= 1.0:
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    
    # Ensure shape is (H, W, 3)
    if rgb_uint8.ndim == 4:
        rgb_uint8 = rgb_uint8.squeeze(0)
    if rgb_uint8.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {rgb_uint8.shape}")
    
    # Convert to PIL Image (SigLIP expects PIL Image)
    pil = Image.fromarray(rgb_uint8, mode="RGB")
    
    # Process with SigLIP processor (handles resizing, normalization, etc.)
    inputs = processor(images=[pil], return_tensors="pt").to(device)
    
    # Encode with SigLIP2
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        emb = out if isinstance(out, torch.Tensor) else out.pooler_output
    
    # Return as numpy array
    emb_np = emb.cpu().numpy().squeeze(0).astype(np.float32)
    
    # Verify embedding dimension matches expected
    if emb_np.shape[0] != RGB_EMBED_DIM:
        raise ValueError(f"Expected RGB embedding dim {RGB_EMBED_DIM}, got {emb_np.shape[0]}")
    
    return emb_np


def encode_depth(depth: np.ndarray, model, device: str, target_size: int = 518, patch_size: int = 14) -> np.ndarray:
    """Encode single depth image (H, W) in meters to (1024,) with DeFM."""
    try:
        from diffusion_policy.dataset.defm_utils import preprocess_depth_batch
    except ImportError:
        from defm_utils import preprocess_depth_batch
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim == 2:
        depth = depth[np.newaxis, :, :]  # (1, H, W)
    else:
        depth = depth[np.newaxis, :, :] if depth.shape[0] != 1 else depth
    normalized = preprocess_depth_batch(
        depth, target_size=target_size, patch_size=patch_size, device=device
    )
    with torch.no_grad():
        out = model.get_intermediate_layers(
            normalized, n=1, reshape=True, return_class_token=True
        )
    class_tok = out[0][1]
    return class_tok.cpu().numpy().squeeze(0).astype(np.float32)


def create_vision_camera(debug_vis: bool = False):
    """Create an Isaac Lab camera (RGB + depth) for vision policy input.
    Camera parent is torso_link from G1 URDF (whole_body_tracking/.../assets/unitree_description/urdf/g1/main.urdf).
    """
    depth_camera: TiledCameraCfg = (
        TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/vision_camera",
            update_period=0.1,  # 10Hz
            height=480,
            width=848,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93,  # D435i: ~87° HFOV
                horizontal_aperture=3.6,
                clipping_range=(0.1, 5.0),
            ),
            debug_vis=debug_vis,
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0, 0.0, 0.0),  # Already positioned by d435_link in URDF
                # pos=(0.35, 0.0, 0.5),  # In front of robot (torso height), meters
                rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention: z-forward
                convention="ros",
            ),
        )
    )
    return depth_camera


def _debug_vision_step(step_count: int, rgb: np.ndarray, depth: np.ndarray, debug_dir: str) -> None:
    """Print vision stats and save RGB/depth images to debug_dir for inspection.
    Depth is visualized with viridis colormap (same as TML-BeyondMimic orig depth):
    close=dark purple, mid=yellow/green, far=blue; invalid/zero depth=white."""
    depth_flat = np.asarray(depth).reshape(-1).astype(np.float64)
    valid = np.isfinite(depth_flat) & (depth_flat > 0)
    depth_min = float(np.min(depth_flat[valid])) if valid.any() else float("nan")
    depth_max = float(np.max(depth_flat[valid])) if valid.any() else float("nan")
    depth_mean = float(np.mean(depth_flat[valid])) if valid.any() else float("nan")
    print(
        f"[VISION step {step_count}] rgb {rgb.shape} dtype={rgb.dtype} | "
        f"depth {np.asarray(depth).shape} min={depth_min:.3f} max={depth_max:.3f} mean={depth_mean:.3f} m"
    )
    os.makedirs(debug_dir, exist_ok=True)
    from PIL import Image

    rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb_uint8.ndim == 4:
        rgb_uint8 = rgb_uint8.squeeze(0)
    Image.fromarray(rgb_uint8, mode="RGB").save(os.path.join(debug_dir, f"step_{step_count:05d}_rgb.png"))
    d = np.asarray(depth).squeeze().astype(np.float64)
    if d.size > 0:
        valid_mask = np.isfinite(d) & (d > 0)
        if valid_mask.any():
            d_min, d_max = float(d[valid_mask].min()), float(d[valid_mask].max())
            # Normalize valid depth to [0, 1]; leave invalid as NaN so colormap maps them to white
            d_norm = np.full_like(d, np.nan, dtype=np.float64)
            d_norm[valid_mask] = (d[valid_mask] - d_min) / (d_max - d_min + 1e-9)
            import matplotlib
            cmap = matplotlib.colormaps["viridis"]
            cmap.set_bad(color="white")
            rgba = cmap(d_norm)  # (H, W, 4) in [0, 1]
            d_vis = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
        else:
            d_vis = np.full((*d.shape, 3), 255, dtype=np.uint8)
        Image.fromarray(d_vis, mode="RGB").save(os.path.join(debug_dir, f"step_{step_count:05d}_depth_vis.png"))
    print(f"[VISION] Saved to {debug_dir}/step_{step_count:05d}_*.png", flush=True)


def get_rgb_depth_from_camera(camera_data, env_id: int = 0):
    """Get RGB (H,W,3) uint8 and depth (H,W) float32 in meters from Isaac camera."""
    rgb_data = camera_data.output["rgb"].detach().cpu().numpy()  # (num_envs, H, W, 3)
    depth_data = camera_data.output["depth"].detach().cpu().numpy()  # (num_envs, H, W, 1)
    return rgb_data[env_id], depth_data[env_id]


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    """Run diffusion policy in Isaac Lab (same flow as sim2sim.py, sim from collect_dataset/play)."""
    print("[INFO] main() entered (Hydra config loaded).", flush=True)
    # Validate checkpoint args (same as sim2sim.py)
    if args_cli.checkpoint is None and args_cli.wandb_path is None:
        args_cli.checkpoint = str(TML_ROOT / "ckpts" / "diffusion_policy_latest.pt")
        print(f"[INFO] Using default checkpoint: {args_cli.checkpoint}")
    if args_cli.checkpoint and args_cli.wandb_path:
        print("[ERROR] Specify only one of --checkpoint or --wandb_path")
        sys.exit(1)

    device = args_cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Create env (play.py / collect_dataset style); use rgb_array for video (works headless)
    record_video = getattr(args_cli, "video", False)
    video_length = getattr(args_cli, "video_length", 500)
    # When recording, use one long episode so the video is one continuous trajectory (no resets)
    if record_video:
        steps_to_seconds = env_cfg.decimation * env_cfg.sim.dt  # e.g. 4 * 0.005 = 0.02s per step
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
    debug_vision = getattr(args_cli, "debug_vision", False)
    # Add depth camera to scene config so it is built with the scene (InteractiveScene does not support item assignment).
    env_cfg.scene.depth_camera = create_vision_camera(debug_vis=debug_vision)
    # Disable debug visuals (contact-force arrows, motion command frames) so they don't appear in the robot's camera view.
    if hasattr(env_cfg.scene, "contact_forces") and hasattr(env_cfg.scene.contact_forces, "debug_vis"):
        env_cfg.scene.contact_forces.debug_vis = False
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "debug_vis"):
        env_cfg.commands.motion.debug_vis = False
    print(f"[INFO] Creating environment (render_mode={render_mode!r}, may take 1-2 min)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode)
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

    # Vision: depth_camera was added to env_cfg.scene before gym.make()
    # Load encoders (SigLIP2 + DeFM)
    print("[INFO] Loading vision encoders (SigLIP2 + DeFM)...", flush=True)
    siglip_model, siglip_processor, defm_model = load_vision_encoders(device)
    print("[INFO] Vision encoders loaded.", flush=True)

    # Guidance (same as sim2sim.py)
    guidance_fn = None
    keyboard_joystick = None
    if args_cli.guidance_type and args_cli.guidance_scale > 0.0:
        guidance_config = {"target_velocity": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
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
    debug_vision_dir = ""
    if debug_vision:
        debug_vision_dir = os.path.abspath(
            os.path.join(getattr(args_cli, "video_folder", "videos/sim2sim_isaaclab"), "vision_debug_dit_action")
        )
        print(f"[VISION] Debug vision enabled: images will be saved to {debug_vision_dir}", flush=True)

    try:
        while step_count < max_steps and simulation_app.is_running():
            # Get state in Isaac order (robot order matches DiffusionAgentIsaac)
            body_pos, body_quat, body_lin_vel, body_ang_vel, joint_pos, joint_vel = _robot_state_isaac(
                env, env_id
            )

            # Update vision camera to follow robot root (needed for valid RGB/depth each step)
            # robot = env.unwrapped.scene["robot"]
            # root_pos = robot.data.root_pos_w[env_id].cpu().unsqueeze(0).to(robot.device)
            # camera_positions = root_pos + torch.tensor([[4.0, 0.0, 2.0]], device=robot.device, dtype=torch.float32)
            # camera_targets = root_pos
            # vision_camera.set_world_poses_from_view(camera_positions, camera_targets)
            # vision_camera.update(dt=env_step_dt)
            
            # Query policy at decimation rate (with vision: encode RGB + depth -> vision_embeds)
            # if step_count % env.unwrapped.cfg.decimation == 0:
            #     camera_data = env.unwrapped.scene["depth_camera"].data
            #     rgb, depth = get_rgb_depth_from_camera(camera_data, env_id)
            #     # Debug: print and save robot vision at selected steps
            #     if debug_vision and debug_vision_dir and (
            #         step_count in (0, 100, 200, 500) or (step_count <= 1000 and step_count > 0 and step_count % 200 == 0)
            #     ):
            #         _debug_vision_step(step_count, rgb, depth, debug_vision_dir)
            #     rgb_emb = encode_rgb(rgb, siglip_model, siglip_processor, device)
            #     depth_emb = encode_depth(depth, defm_model, device)
            #     vision_embeds = np.concatenate([rgb_emb, depth_emb], axis=0).astype(np.float32)
            #     assert vision_embeds.shape[0] == VISION_EMBED_DIM, (
            #         f"Expected vision_embeds dim {VISION_EMBED_DIM}, got {vision_embeds.shape[0]}"
            #     )

            # Query policy every env step (same as play.py; no decimation throttle)
            camera_data = env.unwrapped.scene["depth_camera"].data
            rgb, depth = get_rgb_depth_from_camera(camera_data, env_id)
            # Debug: print and save robot vision at selected steps
            if debug_vision and debug_vision_dir and (
                step_count in (0, 100, 200, 500) or (step_count <= 1000 and step_count > 0 and step_count % 200 == 0)
            ):
                _debug_vision_step(step_count, rgb, depth, debug_vision_dir)
            rgb_emb = encode_rgb(rgb, siglip_model, siglip_processor, device)
            depth_emb = encode_depth(depth, defm_model, device)
            vision_embeds = np.concatenate([rgb_emb, depth_emb], axis=0).astype(np.float32)
            assert vision_embeds.shape[0] == VISION_EMBED_DIM, (
                f"Expected vision_embeds dim {VISION_EMBED_DIM}, got {vision_embeds.shape[0]}"
            )

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
                    vision_embeds=vision_embeds,
                    guidance_fn=guidance_fn,
                    guidance_kwargs=None,
                    guidance_scale=args_cli.guidance_scale,
                )
            else:
                last_action_isaac = policy.get_action(
                    body_pos, body_quat, body_lin_vel, body_ang_vel,
                    joint_pos, joint_vel,
                    # vision_embeds=vision_embeds,
                    vision_embeds=None, # for debugging
                )
            if last_action_isaac is None:
                last_action_isaac = np.zeros(29, dtype=np.float32)
            action_env = _isaac_action_to_env(last_action_isaac, env, env_id)

            # Step env (vec: obs is batched)
            if action_env.shape[0] < env.unwrapped.num_envs:
                action_env = action_env.repeat(env.unwrapped.num_envs, 1)
            obs, _, _, _, _ = env.step(action_env)
            step_count += 1

            if step_count % 500 == 0:
                robot = env.unwrapped.scene["robot"]
                pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
                print(f"Step {step_count}: pelvis height = {pelvis_z:.3f}m")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    finally:
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
