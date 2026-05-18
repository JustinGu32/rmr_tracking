"""
Vision sanity check: does the depth-trained policy walk backwards when an
obstacle appears ~30-40 cm in front of it?

Same scaffolding as sim2sim_isaaclab_vision.py and sim2sim_isaaclab_vision_stairs.py,
but adds a single tall cylinder to the scene and repositions it every control
step so it stays a fixed carrot distance in front of the robot's current
facing direction. If the vision pathway works, the policy should retreat
backwards; the cylinder follows, keeping the stimulus constant.

Usage:
  python scripts/sim2sim_isaaclab_vision_carrot.py \
      --checkpoint /path/to/ckpt.pt \
      --carrot_distance_m 0.35 \
      --headless --video

Tune --carrot_distance_m anywhere in [0.30, 0.40] per the test protocol.
"""

import argparse
import os
import sys
from pathlib import Path
from threading import Lock

import numpy as np
import torch

os.environ["ENABLE_CAMERAS"] = "1"

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

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run diffusion policy in Isaac Lab (G1) with a carrot cylinder in front")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint (.pt)")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path (e.g. user/project/run_id)")
parser.add_argument("--wandb_file", type=str, default="latest.ckpt", help="Checkpoint filename in wandb")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic sampling")
parser.add_argument("--guidance_type", type=str, default=None, help="Guidance type (e.g. joystick, target_heading)")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Isaac Lab task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str, default="/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz", help="Path to motion file for tracking command")
parser.add_argument("--video", action="store_true", help="Record simulation to a video file")
parser.add_argument("--video_folder", type=str, default="videos/vision_carrot", help="Folder to save video")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record")
parser.add_argument("--debug_vision", action="store_true", help="Print and save robot vision (RGB/depth) for debugging")
parser.add_argument("--forward_speed", type=float, default=0.0, help="Forward speed")
parser.add_argument("--lateral_speed", type=float, default=0.0, help="Lateral speed")
parser.add_argument("--spin_speed", type=float, default=0.0, help="Spin speed")
# Carrot cylinder controls
parser.add_argument("--carrot_distance_m", type=float, default=0.35,
                    help="Distance from robot root (xy) to cylinder center, along robot facing direction. "
                         "Test protocol suggests 0.30-0.40 m.")
parser.add_argument("--carrot_radius_m", type=float, default=0.15,
                    help="Cylinder radius in meters.")
parser.add_argument("--carrot_height_m", type=float, default=1.8,
                    help="Cylinder height in meters. Tall enough to show up in head-mounted depth camera view.")
parser.add_argument("--carrot_z_center_m", type=float, default=0.9,
                    help="Cylinder center height above ground (so base of cylinder ~0 if z_center == height/2).")
parser.add_argument("--carrot_warmup_steps", type=int, default=0,
                    help="Skip repositioning the cylinder for this many steps after reset (let it sit far away). "
                         "Useful for an A/B comparison within one episode.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if getattr(args_cli, "video", False):
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac Lab app ready.", flush=True)

import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: E402, F401

sys.path.insert(0, str(TML_ROOT))
from diffusion_policy.inference.guidance import create_guidance_fn  # noqa: E402
from diffusion_policy.inference.diffusion_agent import DiffusionAgentIsaac  # noqa: E402

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    print("[WARNING] pynput not available - keyboard joystick disabled (pip install pynput)")

RGB_EMBED_DIM = 1152
DEPTH_EMBED_DIM = 1024
VISION_EMBED_DIM = RGB_EMBED_DIM + DEPTH_EMBED_DIM

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class KeyboardJoystick:
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


def load_vision_encoders(device: str = "cuda", load_defm: bool = True):
    siglip_model, siglip_processor = None, None
    defm_model = None
    try:
        from transformers import AutoImageProcessor, AutoModel
        ckpt = "google/siglip2-so400m-patch14-384"
        siglip_model = AutoModel.from_pretrained(ckpt, device_map=device).eval()
        siglip_processor = AutoImageProcessor.from_pretrained(ckpt)
        print(f"[VISION] Loaded SigLIP2 from {ckpt}")
    except Exception as e:
        print(f"[VISION] Failed to load SigLIP: {e}")
        raise
    if load_defm:
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
    from PIL import Image
    if rgb.dtype == np.uint8:
        rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    else:
        if rgb.max() <= 1.0:
            rgb_uint8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        else:
            rgb_uint8 = np.clip(rgb, 0, 255).astype(np.uint8)
    if rgb_uint8.ndim == 4:
        rgb_uint8 = rgb_uint8.squeeze(0)
    if rgb_uint8.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got shape {rgb_uint8.shape}")
    pil = Image.fromarray(rgb_uint8, mode="RGB")
    inputs = processor(images=[pil], return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.get_image_features(**inputs)
        emb = out if isinstance(out, torch.Tensor) else out.pooler_output
    emb_np = emb.cpu().numpy().squeeze(0).astype(np.float32)
    if emb_np.shape[0] != RGB_EMBED_DIM:
        raise ValueError(f"Expected RGB embedding dim {RGB_EMBED_DIM}, got {emb_np.shape[0]}")
    return emb_np


def encode_depth(depth: np.ndarray, model, device: str, target_size: int = 518, patch_size: int = 14) -> np.ndarray:
    try:
        from diffusion_policy.dataset.defm_utils import preprocess_depth_batch
    except ImportError:
        from defm_utils import preprocess_depth_batch
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    if depth.ndim == 2:
        depth = depth[np.newaxis, :, :]
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
    np.save(os.path.join(debug_dir, f"step_{step_count:05d}_depth.npy"), np.asarray(depth))
    d = np.asarray(depth).squeeze().astype(np.float64)
    if d.size > 0:
        valid_mask = np.isfinite(d) & (d > 0)
        if valid_mask.any():
            d_min, d_max = float(d[valid_mask].min()), float(d[valid_mask].max())
            d_norm = np.full_like(d, np.nan, dtype=np.float64)
            d_norm[valid_mask] = (d[valid_mask] - d_min) / (d_max - d_min + 1e-9)
            import matplotlib
            cmap = matplotlib.colormaps["viridis"]
            cmap.set_bad(color="white")
            rgba = cmap(d_norm)
            d_vis = (np.clip(rgba[..., :3], 0, 1) * 255).astype(np.uint8)
        else:
            d_vis = np.full((*d.shape, 3), 255, dtype=np.uint8)
        Image.fromarray(d_vis, mode="RGB").save(os.path.join(debug_dir, f"step_{step_count:05d}_depth_vis.png"))
    print(f"[VISION] Saved to {debug_dir}/step_{step_count:05d}_*.png", flush=True)


def _yaw_from_quat_wxyz(w: float, x: float, y: float, z: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def add_carrot_cylinder_to_scene(env_cfg, radius_m: float, height_m: float):
    """Attach a kinematic cylinder to the scene. Spawned far away; repositioned each step."""
    env_cfg.scene.carrot_cylinder = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/CarrotCylinder",
        spawn=sim_utils.CylinderCfg(
            radius=float(radius_m),
            height=float(height_m),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        # Park it far away initially so the reset frame isn't contaminated by a
        # collision or a mid-frame warp before the first reposition call.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1000.0, 1000.0, 500.0)),
    )
    print(f"[SCENE] Carrot cylinder cfg added: radius={radius_m:.3f} m, height={height_m:.3f} m", flush=True)


def reposition_carrot_cylinder(env, distance_m: float, z_center_m: float, device: str, env_id: int = 0):
    """Write the cylinder's root state so it sits `distance_m` in front of the robot along its facing yaw."""
    scene = env.unwrapped.scene
    if "carrot_cylinder" not in scene.keys():
        return
    robot = scene["robot"]
    root_pos_w = robot.data.root_pos_w[env_id].detach().cpu().numpy()
    root_quat_w = robot.data.root_quat_w[env_id].detach().cpu().numpy()  # (w, x, y, z)
    yaw = _yaw_from_quat_wxyz(*root_quat_w)
    fwd = np.array([np.cos(yaw), np.sin(yaw)], dtype=np.float64)

    target_xy = root_pos_w[:2] + fwd * float(distance_m)

    cyl = scene["carrot_cylinder"]
    state = cyl.data.root_state_w[env_id:env_id + 1].clone()
    state[0, 0] = float(target_xy[0])
    state[0, 1] = float(target_xy[1])
    state[0, 2] = float(z_center_m)
    state[0, 3] = 1.0  # w
    state[0, 4] = 0.0
    state[0, 5] = 0.0
    state[0, 6] = 0.0  # identity quat
    state[0, 7:] = 0.0
    cyl.write_root_state_to_sim(state, env_ids=torch.tensor([env_id], device=device))


def _patch_legacy_combined_normalizer(policy) -> None:
    actor = getattr(policy, "actor", None)
    normalizer = getattr(actor, "normalizer", None) if actor is not None else None
    if normalizer is None:
        return
    horizon = int(getattr(actor, "horizon", 0))
    n_past = int(getattr(actor, "n_past_steps", 0))
    if horizon <= 1:
        return
    stream_dims = {
        "obs": int(getattr(actor, "obs_dim", 0)) or None,
        "action": int(getattr(actor, "action_dim", 0)) or None,
    }
    patched_any = False
    for key, expected_D in stream_dims.items():
        if key not in normalizer.params_dict:
            continue
        p = normalizer.params_dict[key]
        scale = p["scale"].data
        if scale.ndim != 1:
            continue
        total = scale.shape[0]
        if expected_D and total == expected_D:
            continue
        if total % horizon != 0:
            continue
        D = total // horizon
        if expected_D and D != expected_D:
            continue
        for k in ("scale", "offset"):
            v = p[k].data
            p[k].data = v.reshape(horizon, D).mean(dim=0)
        input_stats = p["input_stats"]
        for k in ("min", "max", "mean", "std"):
            if k in input_stats:
                v = input_stats[k].data
                if k == "min":
                    input_stats[k].data = v.reshape(horizon, D).min(dim=0).values
                elif k == "max":
                    input_stats[k].data = v.reshape(horizon, D).max(dim=0).values
                else:
                    input_stats[k].data = v.reshape(horizon, D).mean(dim=0)
        patched_any = True
        print(
            f"[INFO] Patched legacy normalizer for key='{key}': "
            f"({horizon}*{D}={total},) -> ({D},) [n_past_steps={n_past}]",
            flush=True,
        )
    if patched_any and hasattr(actor, "set_normalizer"):
        actor.set_normalizer(normalizer)
        print("[INFO] Refreshed actor's cached scale/offset from patched normalizer.", flush=True)
    if not patched_any:
        print("[INFO] Normalizer shapes look clean; no legacy patch applied.", flush=True)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    print("[INFO] main() entered (Hydra config loaded).", flush=True)
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

    add_carrot_cylinder_to_scene(env_cfg, args_cli.carrot_radius_m, args_cli.carrot_height_m)

    record_video = getattr(args_cli, "video", False)
    video_length = getattr(args_cli, "video_length", 500)

    steps_to_seconds = env_cfg.decimation * env_cfg.sim.dt
    episode_s = (max(args_cli.steps, video_length) + 200) * steps_to_seconds
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, episode_s)
    if hasattr(env_cfg.terminations, "anchor_pos") and hasattr(env_cfg.terminations.anchor_pos, "params"):
        env_cfg.terminations.anchor_pos.params["threshold"] = 100.0
    if hasattr(env_cfg.terminations, "anchor_ori") and hasattr(env_cfg.terminations.anchor_ori, "params"):
        env_cfg.terminations.anchor_ori.params["threshold"] = 100.0
    if hasattr(env_cfg.terminations, "ee_body_pos") and hasattr(env_cfg.terminations.ee_body_pos, "params"):
        env_cfg.terminations.ee_body_pos.params["threshold"] = 100.0
    if hasattr(env_cfg.terminations, "bad_anchor_pos_xy") and hasattr(env_cfg.terminations.bad_anchor_pos_xy, "params"):
        env_cfg.terminations.bad_anchor_pos_xy.params["threshold"] = 100.0
    print(f"[INFO] Relaxed termination thresholds for sim2sim (episode_length_s={env_cfg.episode_length_s:.1f})", flush=True)

    render_mode = "rgb_array" if record_video else None
    debug_vision = getattr(args_cli, "debug_vision", False)

    if hasattr(env_cfg.scene, "contact_forces") and hasattr(env_cfg.scene.contact_forces, "debug_vis"):
        env_cfg.scene.contact_forces.debug_vis = False
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "debug_vis"):
        env_cfg.commands.motion.debug_vis = False

    print("[INFO] Loading diffusion policy (DiffusionAgentIsaac); wandb download can be slow...", flush=True)
    model_name = ""
    if args_cli.checkpoint:
        policy = DiffusionAgentIsaac(
            checkpoint_path=args_cli.checkpoint,
            device=device,
            compile=False,
            warmup=False,
            deterministic=args_cli.deterministic,
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
        )
        model_name = args_cli.wandb_path.split("/")[-1]

    _patch_legacy_combined_normalizer(policy)
    print("[INFO] Diffusion policy loaded (Isaac ordering).", flush=True)

    print("[INFO] Loading vision encoders (SigLIP2 + DeFM)...", flush=True)
    siglip_model, siglip_processor, defm_model = load_vision_encoders(device, load_defm=True)
    print("[INFO] Vision encoders loaded.", flush=True)

    print(f"[INFO] Creating environment (render_mode={render_mode!r}, may take 1-2 min)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)
    print("[INFO] Environment created.", flush=True)
    try:
        scene_keys = list(env.unwrapped.scene.keys())
        print(f"[SCENE] Scene entities after env creation: {scene_keys}", flush=True)
        if "carrot_cylinder" not in scene_keys:
            print("[SCENE][ERROR] 'carrot_cylinder' is NOT in the scene — the cfg mutation was lost.", flush=True)
    except Exception as e:
        print(f"[SCENE] Could not list scene keys: {e}", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(getattr(args_cli, "video_folder", "videos/vision_carrot")))
        video_length = getattr(args_cli, "video_length", 500)
        carrot_tag = f"carrot{args_cli.carrot_distance_m:.2f}m"
        if args_cli.guidance_type:
            video_name = f"{model_name}_{carrot_tag}_fwd{args_cli.forward_speed}_lat{args_cli.lateral_speed}_spin{args_cli.spin_speed}_scale{args_cli.guidance_scale}"
        else:
            video_name = f"{model_name}_{carrot_tag}_noguidance"
        os.makedirs(video_folder, exist_ok=True)
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
        print(f"[INFO] Video recording: first {video_length} steps @ {video_fps} FPS -> {video_folder} (prefix: {video_name})")

    guidance_fn = None
    keyboard_joystick = None
    if args_cli.guidance_type and args_cli.guidance_scale > 0.0:
        guidance_config = {
            "dataset_class": "root_only",
            "target_velocity": [args_cli.forward_speed, args_cli.lateral_speed, 0.0, 0.0, 0.0, args_cli.spin_speed],
            "root_vel_indices": (3, 9),
        }
        guidance_fn = create_guidance_fn(args_cli.guidance_type, guidance_config, torch.device(device))
        policy.actor.guidance_inpaint_nominal_state = False
        print(f"[GUIDANCE] {args_cli.guidance_type} scale={args_cli.guidance_scale}")
        if args_cli.guidance_type == "joystick":
            keyboard_joystick = KeyboardJoystick()

    policy.reset()
    print("[INFO] Resetting environment (first step may be slow)...", flush=True)
    obs, _ = env.reset()
    print("[INFO] Environment reset; starting control loop.", flush=True)
    # Seed cylinder position in front of the robot before the first camera frame is used.
    if args_cli.carrot_warmup_steps <= 0:
        reposition_carrot_cylinder(env, args_cli.carrot_distance_m, args_cli.carrot_z_center_m, device)
        print(f"[CARROT] Cylinder placed {args_cli.carrot_distance_m:.3f} m ahead of robot (initial).", flush=True)
    else:
        print(f"[CARROT] Warmup {args_cli.carrot_warmup_steps} steps: cylinder stays parked far away.", flush=True)

    step_count = 0
    max_steps = args_cli.steps
    env_id = 0
    action = None
    debug_vision_dir = ""
    if debug_vision:
        debug_vision_dir = os.path.abspath(
            os.path.join(getattr(args_cli, "video_folder", "videos/vision_carrot"), "vision_debug")
        )
        print(f"[VISION] Debug vision enabled: images will be saved to {debug_vision_dir}", flush=True)

    while step_count < max_steps and simulation_app.is_running():
        # Carrot-on-a-stick: reposition BEFORE reading the camera so the rendered
        # frame for this step already shows the cylinder at distance_m ahead.
        if step_count >= args_cli.carrot_warmup_steps:
            reposition_carrot_cylinder(env, args_cli.carrot_distance_m, args_cli.carrot_z_center_m, device)

        dc = obs['diffusion_collect']
        _idx = env_id if dc['body_pos'].ndim > 1 else slice(None)
        body_pos = dc['body_pos'][_idx].float().cpu().numpy().reshape(30, 3)
        body_quat = dc['body_ori'][_idx].float().cpu().numpy().reshape(30, 4)
        body_lin_vel = dc['body_lin_vel'][_idx].float().cpu().numpy().reshape(30, 3)
        body_ang_vel = dc['body_ang_vel'][_idx].float().cpu().numpy().reshape(30, 3)
        joint_pos = dc['dof_pos'][_idx].float().cpu().numpy()
        joint_vel = dc['dof_vel'][_idx].float().cpu().numpy()

        camera_data = env.unwrapped.scene["depth_camera"].data
        rgb = camera_data.output["rgb"].detach().cpu().numpy()
        depth = camera_data.output["depth"].detach().cpu().numpy()
        if debug_vision and step_count % 200 == 0:
            _debug_vision_step(step_count, rgb, depth, debug_vision_dir)
        rgb_emb = encode_rgb(rgb, siglip_model, siglip_processor, device)
        depth_emb = encode_depth(depth, defm_model, device)
        vision_embeds = np.concatenate([rgb_emb, depth_emb], axis=0).astype(np.float32)

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
                vision_embeds=vision_embeds,
                guidance_fn=guidance_fn,
                guidance_kwargs=None,
                guidance_scale=args_cli.guidance_scale,
            )
        else:
            action = policy.get_action(
                body_pos, body_quat, body_lin_vel, body_ang_vel,
                joint_pos, joint_vel,
                vision_embeds=vision_embeds,
            )
        if action is None:
            action = np.zeros(29, dtype=np.float32)

        action = torch.from_numpy(action).float().to(device).unsqueeze(0)

        if action.shape[0] < env.unwrapped.num_envs:
            action = action.repeat(env.unwrapped.num_envs, 1)
        obs, _, _, _, _ = env.step(action)
        step_count += 1

        if step_count % 100 == 0:
            robot = env.unwrapped.scene["robot"]
            pelvis_xy = robot.data.root_pos_w[env_id, :2].detach().cpu().numpy()
            pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
            print(f"Step {step_count}: root xy = ({pelvis_xy[0]:.3f}, {pelvis_xy[1]:.3f}), pelvis height = {pelvis_z:.3f}m")

    if keyboard_joystick is not None:
        keyboard_joystick.stop()

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
