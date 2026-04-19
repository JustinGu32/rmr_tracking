"""
Run diffusion policy in Isaac Lab (G1 Tracking-Flat) instead of MuJoCo.
Same logic as sim2sim.py: load policy, run control loop with optional guidance,
but simulation is Isaac Lab as in rmr_tracking/scripts/collect_dataset.py and play.py.

Usage (from rmr_tracking or with PYTHONPATH including rmr_tracking):
  python scripts/sim2sim_isaaclab_vision.py --task=Tracking-Flat-G1-v0 --checkpoint ckpts/diffusion_policy_latest.pt
  python scripts/sim2sim_isaaclab_vision.py --task=Tracking-Flat-G1-v0 --wandb_path user/project/run_id

Headless + save video (for servers without display):
  --headless          Disable interactive viewer (required on servers).
  --video             Record simulation to a video file (uses offscreen rendering).
  --video_folder DIR  Where to save the video (default: videos/vision).
  --video_length N    Steps to record (default: 500).
  Example: python scripts/sim2sim_isaaclab_vision.py --task=Tracking-Flat-G1-v0 --checkpoint ckpts/foo.pt --headless --video --video_folder ./out
"""

import argparse
import os
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import torch

# Vision script needs the depth camera; set before any Isaac Lab imports
# (tracking_env_cfg adds depth_camera to scene only when ENABLE_CAMERAS=1)
os.environ["ENABLE_CAMERAS"] = "1"

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
parser.add_argument("--video_folder", type=str, default="videos/vision", help="Folder to save video (default: videos/vision)")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record (default: 500)")
parser.add_argument("--debug_vision", action="store_true", help="Print and save robot vision (RGB/depth) for debugging")
parser.add_argument("--forward_speed", type=float, default=0.0, help="Forward speed")
parser.add_argument("--lateral_speed", type=float, default=0.0, help="Lateral speed")
parser.add_argument("--spin_speed", type=float, default=0.0, help="Spin speed")
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
try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[WARNING] pynput not available - keyboard joystick disabled (pip install pynput)")

# Vision embedding dims: SigLIP2 1152 + DeFM 1024
RGB_EMBED_DIM = 1152
DEPTH_EMBED_DIM = 1024
VISION_EMBED_DIM = RGB_EMBED_DIM + DEPTH_EMBED_DIM  # 2176

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
    if hasattr(env_cfg.terminations, "bad_anchor_pos_xy") and hasattr(env_cfg.terminations.bad_anchor_pos_xy, "params"):                                                                                                                                                  
        env_cfg.terminations.bad_anchor_pos_xy.params["threshold"] = 100.0 
    print(f"[INFO] Relaxed termination thresholds for sim2sim (episode_length_s={env_cfg.episode_length_s:.1f})", flush=True)
    
    render_mode = "rgb_array" if record_video else None
    debug_vision = getattr(args_cli, "debug_vision", False)
    
    # Disable debug visuals (contact-force arrows, motion command frames) so they don't appear in the robot's camera view.
    if hasattr(env_cfg.scene, "contact_forces") and hasattr(env_cfg.scene.contact_forces, "debug_vis"):
        env_cfg.scene.contact_forces.debug_vis = False
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "debug_vis"):
        env_cfg.commands.motion.debug_vis = False

    # Load diffusion policy (Isaac ordering: no MuJoCo conversion inside agent)
    print("[INFO] Loading diffusion policy (DiffusionAgentIsaac); wandb download can be slow...", flush=True)
    model_name = ""
    if args_cli.checkpoint:
        policy = DiffusionAgentIsaac(
            checkpoint_path=args_cli.checkpoint,
            device=device,
            compile=False,
            warmup=False,
            deterministic=args_cli.deterministic,
            # use_two_phase=True,
        )
        model_name = args_cli.checkpoint.split("/")[-3]
    else:
        policy = DiffusionAgentIsaac(
            wandb_path=args_cli.wandb_path,
            checkpoint_file=args_cli.wandb_file,
            device=device,
            compile=False,
            warmup=False,
            deterministic=args_cli.deterministic,
            # use_two_phase=True,
        )
        model_name = args_cli.wandb_path.split("/")[-1]
    print("[INFO] Diffusion policy loaded (Isaac ordering).", flush=True)

    # Load encoders (SigLIP2 + DeFM)
    print("[INFO] Loading vision encoders (SigLIP2 + DeFM)...", flush=True)
    siglip_model, siglip_processor, defm_model = load_vision_encoders(device)
    print("[INFO] Vision encoders loaded.", flush=True)
    
    print(f"[INFO] Creating environment (render_mode={render_mode!r}, may take 1-2 min)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)
    print("[INFO] Environment created.", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(getattr(args_cli, "video_folder", "videos/sim2sim_isaaclab")))
        video_length = getattr(args_cli, "video_length", 500)
        if args_cli.guidance_type:
            video_name = f"{model_name}_forward{args_cli.forward_speed}_lateral{args_cli.lateral_speed}_spin{args_cli.spin_speed}_scale{args_cli.guidance_scale}"
        else:
            video_name = f"{model_name}_noguidance"
        os.makedirs(video_folder, exist_ok=True)
        # Match video FPS to sim control rate (1 step = decimation*dt sec) so playback is smooth.
        # With obstacles, heavier physics can make wall-clock step time variable; explicit fps
        # ensures encoding is consistent and avoids jitter from wrong/default fps.
        video_fps = round(1.0 / (env_cfg.decimation * env_cfg.sim.dt))
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=video_name,
            disable_logger=True,
            fps=video_fps,
        )
        print(f"[INFO] Video recording: first {video_length} steps @ {video_fps} FPS -> {video_folder} (prefix: {video_name} -> {video_name}-step-0.mp4)")
    
    # Guidance
    guidance_fn = None
    keyboard_joystick = None
    if args_cli.guidance_type and args_cli.guidance_scale > 0.0:
        guidance_config = {
            "dataset_class": "root_only",
            "target_velocity": [args_cli.forward_speed, args_cli.lateral_speed, 0.0, 0.0, 0.0, args_cli.spin_speed],
            # FOR REDUCED:
            # "dataset_class": "G1Dataset",
            # "root_pos_indices": (58, 61),
            # "root_vel_indices": (64, 70) # THIS IS FOR REDUCED. when running for limited, just comment this out
            "root_vel_indices": (3, 9) # this is for root only
        }
        guidance_fn = create_guidance_fn(args_cli.guidance_type, guidance_config, torch.device(device))
        # empirically i found that this works better than the default (True) for listening to guidance
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
    action = None
    debug_vision_dir = ""
    if debug_vision:
        debug_vision_dir = os.path.abspath(
            os.path.join(getattr(args_cli, "video_folder", "videos/sim2sim_isaaclab"), "vision_debug")
        )
        print(f"[VISION] Debug vision enabled: images will be saved to {debug_vision_dir}", flush=True)

    while step_count < max_steps and simulation_app.is_running():
        # Get state from diffusion_collect (flattened); reshape to (30, 3), (30, 4) for policy
        dc = obs['diffusion_collect']
        _idx = env_id if dc['body_pos'].ndim > 1 else slice(None)
        body_pos = dc['body_pos'][_idx].float().cpu().numpy().reshape(30, 3)
        body_quat = dc['body_ori'][_idx].float().cpu().numpy().reshape(30, 4)
        body_lin_vel = dc['body_lin_vel'][_idx].float().cpu().numpy().reshape(30, 3)
        body_ang_vel = dc['body_ang_vel'][_idx].float().cpu().numpy().reshape(30, 3)
        joint_pos = dc['dof_pos'][_idx].float().cpu().numpy()
        joint_vel = dc['dof_vel'][_idx].float().cpu().numpy()
        
        # Get vision embeds
        camera_data = env.unwrapped.scene["depth_camera"].data
        rgb = camera_data.output["rgb"].detach().cpu().numpy()
        depth = camera_data.output["depth"].detach().cpu().numpy()
        # Debug: print and save robot vision at selected steps
        if debug_vision and step_count % 200 == 0:
            _debug_vision_step(step_count, rgb, depth, debug_vision_dir)
        rgb_emb = encode_rgb(rgb, siglip_model, siglip_processor, device)
        depth_emb = encode_depth(depth, defm_model, device)
        vision_embeds = np.concatenate([rgb_emb, depth_emb], axis=0).astype(np.float32)
        # assert vision_embeds.shape[0] == VISION_EMBED_DIM, (
        #     f"Expected vision_embeds dim {VISION_EMBED_DIM}, got {vision_embeds.shape[0]}"
        # )

        # Query policy every env step (each env.step() advances decimation physics steps;
        # we need a fresh action per step, matching play.py and collect_dataset)
        if guidance_fn is not None:
            if args_cli.guidance_type == "joystick" and keyboard_joystick is not None:
                if hasattr(guidance_fn, "joystick_values"):
                    guidance_fn.joystick_values[0] = keyboard_joystick.lx
                    guidance_fn.joystick_values[1] = keyboard_joystick.ly
                    guidance_fn.joystick_values[2] = keyboard_joystick.rx
                    guidance_fn.joystick_values[3] = keyboard_joystick.ry
            action = policy.get_action(
                body_pos, body_quat, body_lin_vel, body_ang_vel,
                joint_pos, joint_vel,
                vision_embeds=rgb_emb, # vision_embeds
                guidance_fn=guidance_fn,
                guidance_kwargs=None,
                guidance_scale=args_cli.guidance_scale,
            )
        else:
            action = policy.get_action(
                body_pos, body_quat, body_lin_vel, body_ang_vel,
                joint_pos, joint_vel,
                vision_embeds=rgb_emb, # vision_embeds
                # vision_embeds=None, # for debugging
            )
        if action is None:
            action = np.zeros(29, dtype=np.float32)
            
        action = torch.from_numpy(action).float().to(device).unsqueeze(0)

        # Step env (vec: obs is batched)
        if action.shape[0] < env.unwrapped.num_envs:
            action = action.repeat(env.unwrapped.num_envs, 1)
        obs, _, _, _, _ = env.step(action)
        step_count += 1

        # Print progress
        if step_count % 100 == 0:
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
