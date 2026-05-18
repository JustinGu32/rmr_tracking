"""
sim2sim with the same Isaac Lab scene used by relabel_vision_stairs.py.

Mirrors the collaborator's data-collection scene:
  - task = Tracking-Flat-G1-Collect-v0
  - NUM_OBSTACLES=0 (no pillars)
  - Staircase RigidObjectCfg added to scene using the same URDF, scale, and
    base -90 deg Z rotation as relabel_vision_stairs.py.
  - Depth camera update_period forced to sim.dt (every physics step) so the
    rendered frames match what relabel produced.

Usage:
  python scripts/sim2sim_isaaclab_vision_stairs.py \
      --checkpoint /path/to/ckpt.pt \
      --with_stairs \
      --staircase_ahead_m 2.0 \
      --staircase_yaw_bias_deg 0 \
      --headless --video

The staircase is placed staircase_ahead_m meters in front of the env origin
along +X (robot's default forward). Use --without_stairs to render the
matched no-stairs scene (relabeled_no_stairs_dataset.zarr equivalent).
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from threading import Lock

import numpy as np
import torch

# Cameras + no pillars — must precede any isaaclab imports that read env vars
os.environ["ENABLE_CAMERAS"] = "1"
os.environ["NUM_OBSTACLES"] = "0"

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

parser = argparse.ArgumentParser(description="Run diffusion policy in Isaac Lab (G1) with staircase scene")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint (.pt)")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path (e.g. user/project/run_id)")
parser.add_argument("--wandb_file", type=str, default="latest.ckpt", help="Checkpoint filename in wandb")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic sampling")
parser.add_argument("--guidance_type", type=str, default=None, help="Guidance type (e.g. joystick, target_heading)")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
# Task default matches relabel_vision_stairs.sh (Collect-v0 is what the data was labelled with)
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-Collect-v0", help="Isaac Lab task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str, default="/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz", help="Path to motion file for tracking command")
parser.add_argument("--video", action="store_true", help="Record simulation to a video file")
parser.add_argument("--video_folder", type=str, default="videos/vision_stairs", help="Folder to save video")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record")
parser.add_argument("--debug_vision", action="store_true", help="Print and save robot vision (RGB/depth) for debugging")
parser.add_argument("--no_vision", action="store_true",
                    help="Skip vision encoding and pass vision_embeds=None to the policy. "
                         "Use this when running checkpoints that were trained without vision.")
parser.add_argument("--forward_speed", type=float, default=0.0, help="Forward speed")
parser.add_argument("--lateral_speed", type=float, default=0.0, help="Lateral speed")
parser.add_argument("--spin_speed", type=float, default=0.0, help="Spin speed")
# Staircase scene controls — mirror relabel_vision_stairs.py semantics
parser.add_argument("--with_stairs", dest="with_stairs", action="store_true",
                    help="Insert the staircase into the scene (matches --with_stairs in relabel_vision_stairs.py).")
parser.add_argument("--without_stairs", dest="with_stairs", action="store_false",
                    help="Match the no-stairs relabeled dataset scene (no staircase).")
parser.set_defaults(with_stairs=True)
parser.add_argument("--staircase_ahead_m", type=float, default=2.0,
                    help="Distance along +X from env origin to the first step face. "
                         "relabel used 0.15m from the robot's trajectory end; here the "
                         "robot walks from the origin, so a larger default (~2m) is sensible.")
parser.add_argument("--staircase_yaw_bias_deg", type=float, default=0.0,
                    help="Extra yaw bias in degrees for the staircase (matches relabel arg).")
parser.add_argument("--staircase_lateral_m", type=float, default=0.0,
                    help="Sideways offset of the staircase relative to the robot's facing "
                         "direction, in meters. Positive shifts the staircase to the robot's "
                         "left (perpendicular to facing_dir); negative shifts it right.")
# Control rate — training data was recorded at 50 Hz; the collect env's default is
# decimation=6 * sim.dt=0.005 = ~33 Hz, so override to match the dataset.
parser.add_argument("--control_hz", type=float, default=50.0,
                    help="Control-loop frequency in Hz (1 / (decimation * sim.dt)). "
                         "Training data was collected at 50 Hz; keep this matched.")
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

# Staircase asset constants — must match relabel_vision_stairs.py so the
# rendered stairs look identical to the training data.
STAIRCASE_URDF = "/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase/multi_boxes_scaled_0.84_0.84_0.84.urdf"
STAIRCASE_USD_DIR = os.path.expanduser("~/tmp/IsaacLab/staircase_usd")
STAIRCASE_BOX1_OBJ = Path("/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase/box_models/box1.obj")
STAIRCASE_SCALE = 0.8380952380952381

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def get_first_step_face_offset_m() -> float:
    """Distance from staircase root prim origin to the lowest step face along ascent.
    Copied from relabel_vision_stairs.py so staircase placement matches exactly."""
    default_offset_m = 0.345 * STAIRCASE_SCALE
    try:
        min_y = None
        with open(STAIRCASE_BOX1_OBJ, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    y = float(line.split()[2])
                    min_y = y if min_y is None else min(min_y, y)
        if min_y is None:
            raise ValueError("No vertices found in staircase mesh.")
        return abs(min_y) * STAIRCASE_SCALE
    except (OSError, ValueError, IndexError):
        warnings.warn(
            f"Falling back to hard-coded first-step offset {default_offset_m:.4f} m",
            stacklevel=2,
        )
        return default_offset_m


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


def load_vision_encoders(device: str = "cuda"):
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


def compute_staircase_pose(staircase_ahead_m: float, yaw_bias_deg: float, lateral_m: float = 0.0):
    """Return (pos_xyz, quat_wxyz) for the staircase root in env-local coords.

    The robot spawns at the env origin facing +X, so we place the first step
    face at +X = staircase_ahead_m and apply the same base -90 deg Z rotation
    as relabel so stairs ascend along +X toward the approaching robot. A
    positive lateral_m shifts the staircase along +Y (robot's left)."""
    first_step_offset = get_first_step_face_offset_m()
    stair_root_x = float(staircase_ahead_m) + first_step_offset
    total_yaw = -np.pi / 2.0 + np.deg2rad(yaw_bias_deg)
    half = total_yaw / 2.0
    pos = (stair_root_x, float(lateral_m), 0.0)
    quat_wxyz = (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))
    return pos, quat_wxyz, first_step_offset


def add_staircase_to_scene(env_cfg, staircase_ahead_m: float, yaw_bias_deg: float, lateral_m: float = 0.0):
    """Attach a staircase RigidObject to env_cfg.scene matching relabel_vision_stairs.py."""
    pos, rot_wxyz, first_step_offset = compute_staircase_pose(staircase_ahead_m, yaw_bias_deg, lateral_m)
    env_cfg.scene.staircase = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Staircase",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=STAIRCASE_URDF,
            usd_dir=STAIRCASE_USD_DIR,
            fix_base=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot_wxyz),
    )
    print(f"[SCENE] Staircase cfg added: root at (+X={pos[0]:.3f}, +Y={pos[1]:.3f}, 0), "
          f"first-step face at +X={staircase_ahead_m:.3f}, lateral={lateral_m:.3f} m, "
          f"yaw_bias={yaw_bias_deg:.1f} deg, first_step_offset={first_step_offset:.3f} m", flush=True)


def _yaw_from_quat_wxyz(w: float, x: float, y: float, z: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def reposition_staircase_in_sim(env, staircase_ahead_m: float, yaw_bias_deg: float, device: str, lateral_m: float = 0.0):
    """Write the staircase root state into sim, placing it ahead of the robot.

    In a tracking env the robot does not spawn at (0, 0) — it spawns wherever
    the motion file's first anchor frame puts it and walks in that frame's
    heading direction. So we read the robot's current world pose and put the
    staircase (first step face) staircase_ahead_m meters in front of it, with
    the base -90 deg Z rotation (so URDF +Y ascent becomes +forward ascent)
    plus the optional yaw bias — the same formula relabel_vision_stairs.py
    uses with the trajectory's final heading."""
    scene = env.unwrapped.scene
    if "staircase" not in scene.keys():
        print(f"[SCENE][WARN] 'staircase' not in scene.keys(); keys={list(scene.keys())}", flush=True)
        return

    robot = scene["robot"]
    root_pos_w = robot.data.root_pos_w[0].detach().cpu().numpy()        # world xyz
    root_quat_w = robot.data.root_quat_w[0].detach().cpu().numpy()      # (w, x, y, z)
    facing_yaw = _yaw_from_quat_wxyz(*root_quat_w)
    facing_dir = np.array([np.cos(facing_yaw), np.sin(facing_yaw)], dtype=np.float64)
    # Perpendicular to facing_dir, pointing to the robot's left (90 deg CCW).
    lateral_dir = np.array([-np.sin(facing_yaw), np.cos(facing_yaw)], dtype=np.float64)

    first_step_offset = get_first_step_face_offset_m()
    # first step face in front of robot; staircase root is further along ascent
    first_step_xy = (root_pos_w[:2]
                     + facing_dir * float(staircase_ahead_m)
                     + lateral_dir * float(lateral_m))
    stair_root_xy = first_step_xy + facing_dir * first_step_offset

    total_yaw = facing_yaw - np.pi / 2.0 + np.deg2rad(yaw_bias_deg)
    half = total_yaw / 2.0
    quat_wxyz = (float(np.cos(half)), 0.0, 0.0, float(np.sin(half)))

    staircase_asset = scene["staircase"]
    stair_state = staircase_asset.data.root_state_w[0:1].clone()
    stair_state[0, 0] = float(stair_root_xy[0])
    stair_state[0, 1] = float(stair_root_xy[1])
    stair_state[0, 2] = 0.0  # sit on the ground plane
    stair_state[0, 3] = quat_wxyz[0]
    stair_state[0, 4] = quat_wxyz[1]
    stair_state[0, 5] = quat_wxyz[2]
    stair_state[0, 6] = quat_wxyz[3]
    stair_state[0, 7:] = 0.0
    staircase_asset.write_root_state_to_sim(
        stair_state, env_ids=torch.tensor([0], device=device)
    )
    print(
        f"[SCENE] Robot at world ({root_pos_w[0]:.3f}, {root_pos_w[1]:.3f}, {root_pos_w[2]:.3f}), "
        f"yaw={np.degrees(facing_yaw):.1f} deg. "
        f"First step face at world ({first_step_xy[0]:.3f}, {first_step_xy[1]:.3f}); "
        f"staircase root at world ({stair_root_xy[0]:.3f}, {stair_root_xy[1]:.3f}).",
        flush=True,
    )


def _patch_legacy_combined_normalizer(policy) -> None:
    """Fix ``obs``/``action``/``cond`` normalizer stats saved by the buggy
    ``CombinedDataset.get_normalizer`` (pre-fix), which fit stats on
    ``(N, H*D)`` and so persisted ``scale`` of shape ``(H*D,)`` instead of
    ``(D,)``. At inference we feed ``(B, n_past_steps, D)`` which fails the
    ``reshape(-1, H*D)`` inside ``LinearNormalizer._normalize``. Collapse the
    per-timestep stats back to a single ``(D,)`` by averaging across H.
    Idempotent for already-correct checkpoints.
    """
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
            continue  # already correct
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
        # ``DiffusionActor`` caches ``obs_scale``/``obs_offset``/``action_scale``/
        # ``action_offset`` as cloned tensors at load time for a torch.compile /
        # JIT-friendly fast path (see ``set_normalizer``). Those caches still hold
        # the buggy ``(H*D,)`` shapes, so re-extract them from the now-patched
        # ``normalizer.params_dict`` so ``unnormalize_action`` uses the fixed stats.
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

    # Align control rate with the 50 Hz dataset. Keep sim.dt=0.005 (the collect
    # env's default physics step) and set decimation so decimation*sim.dt == 1/control_hz.
    target_dt = 1.0 / float(args_cli.control_hz)
    new_decimation = max(1, int(round(target_dt / env_cfg.sim.dt)))
    achieved_hz = 1.0 / (new_decimation * env_cfg.sim.dt)
    if abs(achieved_hz - args_cli.control_hz) > 1e-3:
        print(f"[WARN] Could not hit exactly {args_cli.control_hz:.2f} Hz with sim.dt={env_cfg.sim.dt}; "
              f"using decimation={new_decimation} -> {achieved_hz:.3f} Hz", flush=True)
    env_cfg.decimation = new_decimation
    env_cfg.sim.render_interval = env_cfg.decimation
    print(f"[INFO] Control rate: decimation={env_cfg.decimation} * sim.dt={env_cfg.sim.dt} "
          f"-> {achieved_hz:.3f} Hz (target {args_cli.control_hz} Hz)", flush=True)

    # Match relabel: force camera refresh every physics step so depth/RGB for the
    # policy is as fresh as the training-time renders.
    if getattr(env_cfg.scene, "depth_camera", None) is not None:
        env_cfg.scene.depth_camera.update_period = env_cfg.sim.dt

    # Insert staircase into the scene (before env instantiation) to match the
    # collaborator's data-collection scene.
    if args_cli.with_stairs:
        add_staircase_to_scene(env_cfg, args_cli.staircase_ahead_m, args_cli.staircase_yaw_bias_deg,
                               args_cli.staircase_lateral_m)
    else:
        print("[SCENE] --without_stairs: no staircase added (matches no-stairs relabeled dataset).")

    record_video = getattr(args_cli, "video", False)
    video_length = getattr(args_cli, "video_length", 500)

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

    if args_cli.no_vision:
        siglip_model = siglip_processor = defm_model = None
        print("[INFO] --no_vision: skipping vision encoder load; will pass vision_embeds=None.", flush=True)
    else:
        print("[INFO] Loading vision encoders (SigLIP2 + DeFM)...", flush=True)
        siglip_model, siglip_processor, defm_model = load_vision_encoders(device)
        print("[INFO] Vision encoders loaded.", flush=True)

    print(f"[INFO] Creating environment (render_mode={render_mode!r}, may take 1-2 min)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)
    print("[INFO] Environment created.", flush=True)
    try:
        scene_keys = list(env.unwrapped.scene.keys())
        print(f"[SCENE] Scene entities after env creation: {scene_keys}", flush=True)
        if args_cli.with_stairs and "staircase" not in scene_keys:
            print("[SCENE][ERROR] 'staircase' is NOT in the scene — the cfg mutation was lost. "
                  "Check that add_staircase_to_scene ran before gym.make.", flush=True)
    except Exception as e:
        print(f"[SCENE] Could not list scene keys: {e}", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(getattr(args_cli, "video_folder", "videos/vision_stairs")))
        video_length = getattr(args_cli, "video_length", 500)
        stairs_tag = f"stairs_distance{args_cli.staircase_ahead_m}_yaw{args_cli.staircase_yaw_bias_deg}_lat{args_cli.staircase_lateral_m}" if args_cli.with_stairs else "nostairs"
        if args_cli.guidance_type:
            video_name = f"{model_name}_{stairs_tag}_fwd{args_cli.forward_speed}_lat{args_cli.lateral_speed}_spin{args_cli.spin_speed}_scale{args_cli.guidance_scale}"
        else:
            video_name = f"{model_name}_{stairs_tag}_noguidance"
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
    # Kinematic fixed-base rigid objects don't always pick up init_state on reset,
    # so explicitly write the staircase pose into sim (same pattern as relabel).
    if args_cli.with_stairs:
        reposition_staircase_in_sim(env, args_cli.staircase_ahead_m,
                                    args_cli.staircase_yaw_bias_deg, device,
                                    lateral_m=args_cli.staircase_lateral_m)
    step_count = 0
    max_steps = args_cli.steps
    env_id = 0
    action = None
    debug_vision_dir = ""
    if debug_vision:
        debug_vision_dir = os.path.abspath(
            os.path.join(getattr(args_cli, "video_folder", "videos/vision_stairs"), "vision_debug")
        )
        print(f"[VISION] Debug vision enabled: images will be saved to {debug_vision_dir}", flush=True)

    while step_count < max_steps and simulation_app.is_running():
        dc = obs['diffusion_collect']
        _idx = env_id if dc['body_pos'].ndim > 1 else slice(None)
        body_pos = dc['body_pos'][_idx].float().cpu().numpy().reshape(30, 3)
        body_quat = dc['body_ori'][_idx].float().cpu().numpy().reshape(30, 4)
        body_lin_vel = dc['body_lin_vel'][_idx].float().cpu().numpy().reshape(30, 3)
        body_ang_vel = dc['body_ang_vel'][_idx].float().cpu().numpy().reshape(30, 3)
        joint_pos = dc['dof_pos'][_idx].float().cpu().numpy()
        joint_vel = dc['dof_vel'][_idx].float().cpu().numpy()

        if args_cli.no_vision:
            vision_embeds = None
        else:
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
            pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
            print(f"Step {step_count}: pelvis height = {pelvis_z:.3f}m")

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
