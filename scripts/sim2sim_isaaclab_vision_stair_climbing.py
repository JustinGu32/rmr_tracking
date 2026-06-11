"""
sim2sim inference for a policy trained on the collaborator's staircase dataset.

The dataset was generated using the Staircase task (see
`whole_body_tracking/tasks/staircase/staircase_env_cfg.py`). That scene already
includes the staircase as a RigidObjectCfg at the env origin — unlike the old
sim2sim_isaaclab_vision_stairs.py, which took the flat-ground Tracking-Flat-G1
env and manually attached a staircase ahead of each trajectory. Here we just
use the Staircase task directly, so the stairs render exactly as they did
during data collection.

IMPORTANT — match the data-collection vision setup. run_dataset.sh collected
with `--disable_rgb`, so the dataset stores only DeFM depth embeddings (1024-dim)
and the model expects depth-only vision. Pass `--disable_rgb` here too; otherwise
this script feeds the RGB+depth concat (2176-dim) the model was never trained on
(and the dim check will abort).

Usage:
  # depth-only, matching run_dataset.sh (recommended for this dataset):
  python scripts/sim2sim_isaaclab_vision_stair_climbing.py \
      --checkpoint /path/to/ckpt.pt \
      --disable_rgb \
      --headless --video

  # pick a different staircase motion clip:
  python scripts/sim2sim_isaaclab_vision_stair_climbing.py \
      --checkpoint /path/to/ckpt.pt \
      --motion_file /move/u/justingu/whole_body_tracking/artifacts/chair_step_Skeleton_004:v0/motion.npz \
      --headless --video
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path
from threading import Lock

import numpy as np
import torch

# Depth camera must be configured before any isaaclab imports.
# ENABLE_CAMERAS alone is NOT enough for the staircase task: StaircaseEnvCfg.__post_init__
# nulls scene.depth_camera unless WBT_USE_DEPTH_OBS=1 (see staircase_env_cfg.py), which
# would crash at scene["depth_camera"] below. This vision script always needs the camera.
os.environ["ENABLE_CAMERAS"] = "1"
os.environ["WBT_USE_DEPTH_OBS"] = "1"

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

parser = argparse.ArgumentParser(description="Run diffusion policy in Isaac Lab (G1) on the Staircase task")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint (.pt)")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path (e.g. user/project/run_id)")
parser.add_argument("--wandb_file", type=str, default="latest.ckpt", help="Checkpoint filename in wandb")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic sampling")
parser.add_argument("--guidance_type", type=str, default=None, help="Guidance type (e.g. joystick, target_heading)")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
# Staircase-G1-Collect-v0 registers the diffusion_collect observation group the
# policy was trained on. Use -Play-v0 for cleaner viewer playback (no perturbations).
parser.add_argument("--task", type=str, default="Staircase-G1-Collect-v0",
                    help="Isaac Lab task (Staircase-G1-Collect-v0, Staircase-G1-Play-v0, ...)")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str,
                    default="/move/u/chrzhang/rmr_tracking/artifacts/walk_up_33.npz:v0/motion.npz",
                    help="Path to motion file. Must match the staircase MotionLoader format "
                         "(keys: fps, joint_pos, joint_vel, body_pos_w, body_quat_w, "
                         "body_lin_vel_w, body_ang_vel_w; optional object_pos_w).")
parser.add_argument("--video", action="store_true", help="Record simulation to a video file")
parser.add_argument("--video_folder", type=str, default="videos/vision_stair_climbing", help="Folder to save video")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record")
parser.add_argument("--video_name", type=str, default=None, help="Video name prefix (overrides the auto-generated name)")
parser.add_argument("--debug_vision", action="store_true", help="Print and save robot vision (RGB/depth) for debugging")
parser.add_argument("--no_vision", action="store_true",
                    help="Skip vision encoding and pass vision_embeds=None to the policy. "
                         "Use this when running checkpoints that were trained without vision.")
parser.add_argument("--disable_rgb", action="store_true",
                    help="Depth-only vision: skip the SigLIP RGB encoder and pass only the DeFM "
                         "depth embedding (1024-dim) to the policy. MUST match data collection — "
                         "run_dataset.sh collected with --disable_rgb, so the dataset has no "
                         "rgb_embed and the model expects depth-only (1024) vision, NOT the "
                         "RGB+depth (2176) concat this script uses by default.")
parser.add_argument("--forward_speed", type=float, default=0.0, help="Forward speed")
parser.add_argument("--lateral_speed", type=float, default=0.0, help="Lateral speed")
parser.add_argument("--spin_speed", type=float, default=0.0, help="Spin speed")
# Optional control-rate override. The Staircase-Collect env runs decimation=6 *
# sim.dt=0.005 = ~33.33 Hz natively; leave control_hz=None to use that (matches
# the dataset). Set to e.g. 50.0 if your dataset was actually 50 Hz.
parser.add_argument("--control_hz", type=float, default=None,
                    help="Override control-loop frequency in Hz. Default: use the env's native rate "
                         "(Staircase-Collect = decimation 6 * sim.dt 0.005 -> 33.3 Hz, matching 33 Hz "
                         "collection). Pass --control_hz 50 to eval a model trained on the OLD 50 Hz data.")
parser.add_argument("--action_shift", type=int, default=0,
                    help="Inference-latency compensation. Delay the APPLIED action by N control "
                         "steps: the action computed from state s_t is held and applied N steps "
                         "later, at s_{t+N}. MUST match the dataset's action_shift used at training "
                         "(use_shifted_action=true, action_shift=1 in the dit_backbone_vision_root "
                         "config). This sim otherwise applies a_t at s_t (zero latency), whereas "
                         "holosoma/real apply it ~25 ms later at s_{t+1}; the shifted policy predicts "
                         "the t+1 action, so delaying it here makes sim match real. Default 0 (off, "
                         "for checkpoints trained without the shift).")
parser.add_argument("--min_sample_idx", type=int, default=110,
                    help="Min reference-motion frame the robot is reset to.")
parser.add_argument("--max_sample_idx", type=int, default=130,
                    help="Max reference-motion frame the robot is reset to.")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if getattr(args_cli, "video", False):
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
print("[INFO] Isaac Lab app ready.", flush=True)

import gymnasium as gym

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

import whole_body_tracking.tasks  # noqa: E402, F401  — registers Staircase-G1-* tasks

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


def load_vision_encoders(device: str = "cuda", load_rgb: bool = True):
    siglip_model, siglip_processor = None, None
    defm_model = None
    if load_rgb:
        try:
            from transformers import AutoImageProcessor, AutoModel
            ckpt = "google/siglip2-so400m-patch14-384"
            siglip_model = AutoModel.from_pretrained(ckpt, device_map=device).eval()
            siglip_processor = AutoImageProcessor.from_pretrained(ckpt)
            print(f"[VISION] Loaded SigLIP2 from {ckpt}")
        except Exception as e:
            print(f"[VISION] Failed to load SigLIP: {e}")
            raise
    else:
        print("[VISION] --disable_rgb: skipping SigLIP RGB encoder (depth-only, matches collection).")
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
    rgb_shape = None if rgb is None else rgb.shape
    print(
        f"[VISION step {step_count}] rgb {rgb_shape} | "
        f"depth {np.asarray(depth).shape} min={depth_min:.3f} max={depth_max:.3f} mean={depth_mean:.3f} m"
    )
    os.makedirs(debug_dir, exist_ok=True)
    from PIL import Image
    if rgb is not None:
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

    # Optional control-rate override — leave at native rate unless overridden.
    if args_cli.control_hz is not None:
        target_dt = 1.0 / float(args_cli.control_hz)
        new_decimation = max(1, int(round(target_dt / env_cfg.sim.dt)))
        achieved_hz = 1.0 / (new_decimation * env_cfg.sim.dt)
        if abs(achieved_hz - args_cli.control_hz) > 1e-3:
            print(f"[WARN] Could not hit exactly {args_cli.control_hz:.2f} Hz with sim.dt={env_cfg.sim.dt}; "
                  f"using decimation={new_decimation} -> {achieved_hz:.3f} Hz", flush=True)
        env_cfg.decimation = new_decimation
        env_cfg.sim.render_interval = env_cfg.decimation
    native_hz = 1.0 / (env_cfg.decimation * env_cfg.sim.dt)
    print(f"[INFO] Control rate: decimation={env_cfg.decimation} * sim.dt={env_cfg.sim.dt} "
          f"-> {native_hz:.3f} Hz", flush=True)

    # Match the actuator delay used during collection. collect_dataset.py forced
    # min_delay=max_delay=0 for every actuator group; the G1 cfg default is 0..3,
    # so without this the policy would face random latency it never trained on.
    actuators = getattr(getattr(env_cfg.scene, "robot", None), "actuators", None)
    if actuators is not None:
        for act_name, act_cfg in actuators.items():
            if hasattr(act_cfg, "min_delay"):
                act_cfg.min_delay = 0
            if hasattr(act_cfg, "max_delay"):
                act_cfg.max_delay = args_cli.actuator_max_delay
        print(f"[INFO] Actuator delay set to min=0,max={args_cli.actuator_max_delay} for "
              f"{list(actuators.keys())} (collection used 0).", flush=True)

    # Match the depth-camera refresh rate the policy was TRAINED on. Collection
    # left update_period at the cfg default 0.1 s (10 Hz), so the depth embedding
    # only changed every ~5 control steps. Refreshing every physics step (sim.dt)
    # would feed the policy fresher depth than it ever saw, an eval/train mismatch.
    if getattr(env_cfg.scene, "depth_camera", None) is not None:
        env_cfg.scene.depth_camera.update_period = args_cli.camera_update_period
        print(f"[INFO] depth_camera.update_period = {args_cli.camera_update_period}s "
              f"(collection used 0.1s / 10 Hz).", flush=True)
    else:
        print("[WARN] env_cfg.scene.depth_camera is None — was ENABLE_CAMERAS=1 set before imports? "
              "The script set it at module load; if you're running via a wrapper that strips env, "
              "export ENABLE_CAMERAS=1 in your shell.", flush=True)

    record_video = getattr(args_cli, "video", False)
    video_length = getattr(args_cli, "video_length", 500)

    steps_to_seconds = env_cfg.decimation * env_cfg.sim.dt
    episode_s = (max(args_cli.steps, video_length) + 200) * steps_to_seconds
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, episode_s)
    # Relax reference-motion terminations — the diffusion policy doesn't track
    # the reference motion, so these would fire spurious resets during rollout.
    for name, attr in [
        ("anchor_pos_z", "threshold"),
        ("anchor_pos_xy", "threshold"),
        ("anchor_ori", "threshold"),
        ("ee_body_pos", "threshold"),
    ]:
        term = getattr(env_cfg.terminations, name, None)
        if term is not None and hasattr(term, "params") and attr in term.params:
            term.params[attr] = 100.0
    print(f"[INFO] Relaxed termination thresholds for sim2sim (episode_length_s={env_cfg.episode_length_s:.1f})",
          flush=True)

    render_mode = "rgb_array" if record_video else None
    debug_vision = getattr(args_cli, "debug_vision", False)

    if hasattr(env_cfg.scene, "contact_forces") and hasattr(env_cfg.scene.contact_forces, "debug_vis"):
        env_cfg.scene.contact_forces.debug_vis = False
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion") and hasattr(env_cfg.commands.motion, "debug_vis"):
        env_cfg.commands.motion.debug_vis = False

    # The Staircase-Collect env sets DiffusionCollect.concatenate_terms=True, so
    # obs['diffusion_collect'] would be a flat tensor and `dc['body_pos']` would
    # iterate the string into per-char indices -> "too many indices for tensor
    # of dimension 2". Force dict form so dc['body_pos'] etc. work as expected.
    dc_group = getattr(env_cfg.observations, "diffusion_collect", None)
    if dc_group is not None and hasattr(dc_group, "concatenate_terms"):
        dc_group.concatenate_terms = False

    # Disable events that assume we're tracking the reference motion. The
    # diffusion policy generates its own motion, so:
    #   - assistive_spring_force yanks the robot at 600 N/m toward the reference
    #     anchor (which advances along the staircase regardless of what our
    #     robot does). This is the most likely cause of "flies off and zooms
    #     across the scene": the spring drags the robot through the stairs /
    #     into free space as the reference clip plays out.
    #   - push_robot adds random velocity perturbations — fine during data
    #     collection, just adds noise to rollouts.
    # This mirrors what Staircase-G1-Play-v0 does (staircase_env_cfg.py cfg g1).
    if hasattr(env_cfg, "events"):
        if getattr(env_cfg.events, "assistive_spring_force", None) is not None:
            if args_cli.keep_spring:
                print("[INFO] --keep_spring: leaving events.assistive_spring_force ACTIVE "
                      "(matches data collection, where curriculum_factor stayed at 1.0).", flush=True)
            else:
                env_cfg.events.assistive_spring_force = None
                print("[INFO] Disabled events.assistive_spring_force for sim2sim.", flush=True)
        if getattr(env_cfg.events, "push_robot", None) is not None:
            env_cfg.events.push_robot = None
            print("[INFO] Disabled events.push_robot for sim2sim.", flush=True)
    # The curriculum modifies the spring-force factor; with the event gone,
    # the curriculum term would error looking for events.assistive_spring_force.
    if getattr(env_cfg, "curriculum", None) is not None:
        for curr_name in ("spring_force_linear", "spring_force_factor"):
            if getattr(env_cfg.curriculum, curr_name, None) is not None:
                setattr(env_cfg.curriculum, curr_name, None)
                print(f"[INFO] Disabled curriculum.{curr_name}.", flush=True)

    # Reset the robot to reference-motion frames in [min_sample_idx, max_sample_idx],
    # matching collection (140..150) so eval covers the same start-pose distribution
    # the model trained on (just before the stair climb). Set --min_sample_idx ==
    # --max_sample_idx (e.g. both 140) for a deterministic start / clean video.
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        if hasattr(env_cfg.commands.motion, "min_sample_idx"):
            env_cfg.commands.motion.min_sample_idx = args_cli.min_sample_idx
        if hasattr(env_cfg.commands.motion, "max_sample_idx"):
            env_cfg.commands.motion.max_sample_idx = args_cli.max_sample_idx
        print(f"[INFO] Reset sample_idx range = [{args_cli.min_sample_idx}, "
              f"{args_cli.max_sample_idx}] (collection uses [140, 150]).", flush=True)

    print("[INFO] Loading diffusion policy (DiffusionAgentIsaac); wandb download can be slow...", flush=True)
    stabilizer_kwargs = dict(
        enable_action_inpainting=args_cli.action_inpainting,
        temporal_ensemble_size=args_cli.temporal_ensemble_size,
        ensemble_size=args_cli.ensemble_size,
        enable_rolling=args_cli.enable_rolling,
    )
    print(f"[INFO] Closed-loop stabilizers: action_inpainting={args_cli.action_inpainting}, "
          f"temporal_ensemble_size={args_cli.temporal_ensemble_size}, "
          f"ensemble_size={args_cli.ensemble_size}, enable_rolling={args_cli.enable_rolling}", flush=True)
    model_name = ""
    if args_cli.checkpoint:
        policy = DiffusionAgentIsaac(
            checkpoint_path=args_cli.checkpoint,
            device=device,
            compile=False,
            warmup=False,
            deterministic=args_cli.deterministic,
            **stabilizer_kwargs,
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
            **stabilizer_kwargs,
        )
        model_name = args_cli.wandb_path.split("/")[-1]
    print("[INFO] Diffusion policy loaded (Isaac ordering).", flush=True)

    if args_cli.no_vision:
        siglip_model = siglip_processor = defm_model = None
        print("[INFO] --no_vision: skipping vision encoder load; will pass vision_embeds=None.", flush=True)
    else:
        load_rgb = not args_cli.disable_rgb
        enc_desc = "DeFM (depth only)" if args_cli.disable_rgb else "SigLIP2 + DeFM"
        print(f"[INFO] Loading vision encoders ({enc_desc})...", flush=True)
        siglip_model, siglip_processor, defm_model = load_vision_encoders(device, load_rgb=load_rgb)
        print("[INFO] Vision encoders loaded.", flush=True)

    # Sanity-check that the vision embedding we will feed matches what the policy's
    # backbone expects. This is exactly the RGB-vs-depth-only trap: a depth-only
    # model expects 1024 dims, the RGB+depth concat is 2176. Catch it before the
    # control loop instead of failing deep inside the first inference.
    if not args_cli.no_vision:
        expected_vision_dim = None
        backbone = getattr(policy.actor, "backbone", None)
        if backbone is not None:
            for attr in ("vision_input_dim", "v_input_dim", "v_output_dim"):
                val = getattr(backbone, attr, None)
                if val:
                    expected_vision_dim = int(val)
                    break
            # ContextConditionedDiT doesn't expose a vision-dim attribute; read it
            # off the vision context projection layer instead (Linear(vision_dim, hidden)).
            if expected_vision_dim is None:
                proj = getattr(backbone, "vision_ctx_proj1", None)
                if proj is not None and hasattr(proj, "in_features"):
                    expected_vision_dim = int(proj.in_features)
        provided_vision_dim = DEPTH_EMBED_DIM if args_cli.disable_rgb else VISION_EMBED_DIM
        print(f"[VISION] Feeding {provided_vision_dim}-dim embeddings "
              f"({'depth-only' if args_cli.disable_rgb else 'RGB+depth concat'}); "
              f"model expects {expected_vision_dim}.", flush=True)
        if expected_vision_dim is not None and expected_vision_dim != provided_vision_dim:
            raise SystemExit(
                f"[VISION][ERROR] Vision dim mismatch: model expects {expected_vision_dim} but this "
                f"run feeds {provided_vision_dim}. Your dataset was collected with --disable_rgb "
                f"(depth-only, 1024). {'Add' if not args_cli.disable_rgb else 'Remove'} --disable_rgb "
                f"to match."
            )

    print(f"[INFO] Creating environment (render_mode={render_mode!r}, may take 1-2 min)...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)
    print("[INFO] Environment created.", flush=True)
    try:
        scene_keys = list(env.unwrapped.scene.keys())
        print(f"[SCENE] Scene entities after env creation: {scene_keys}", flush=True)
        if "staircase" not in scene_keys:
            print("[SCENE][ERROR] 'staircase' is NOT in the scene — the Staircase task's scene cfg "
                  "should have included it. Check that args_cli.task is a Staircase-G1-* task.", flush=True)
        if "depth_camera" not in scene_keys:
            print("[SCENE][ERROR] 'depth_camera' is NOT in the scene — ENABLE_CAMERAS was likely "
                  "not set before the isaaclab imports.", flush=True)
    except Exception as e:
        print(f"[SCENE] Could not list scene keys: {e}", flush=True)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(getattr(args_cli, "video_folder", "videos/vision_staircase")))
        video_length = getattr(args_cli, "video_length", 500)
        if getattr(args_cli, "video_name", None):
            video_name = args_cli.video_name
        elif args_cli.guidance_type:
            video_name = f"{model_name}_staircase_fwd{args_cli.forward_speed}_lat{args_cli.lateral_speed}_spin{args_cli.spin_speed}_scale{args_cli.guidance_scale}"
        else:
            video_name = f"{model_name}_staircase_noguidance"
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

    step_count = 0
    max_steps = args_cli.steps
    env_id = 0
    action = None
    # Inference-latency compensation: queue of the last `action_shift` predictions.
    # Each step we apply the oldest (computed action_shift steps ago) and push the
    # newest, so a_t computed from s_t is applied at s_{t+action_shift}, matching
    # holosoma/real and the action_shift relabeling used at training time.
    action_queue = None
    if args_cli.action_shift > 0:
        print(f"[INFO] action_shift={args_cli.action_shift}: applied action is delayed "
              f"{args_cli.action_shift} control step(s) to match training relabeling.", flush=True)

    replay_acts = None
    if args_cli.replay_actions:
        import zarr as _zarr
        _z = _zarr.open(args_cli.replay_zarr, "r")
        _ee = _z["meta/episode_ends"][:]
        _starts = np.concatenate([[0], _ee[:-1]])
        _s, _e = int(_starts[args_cli.replay_episode]), int(_ee[args_cli.replay_episode])
        replay_acts = _z["data/act"][_s:_e].astype(np.float32)
        print(f"[REPLAY] Open-loop replay of recorded data/act ep={args_cli.replay_episode} "
              f"({replay_acts.shape[0]} frames) from {args_cli.replay_zarr}. Policy is BYPASSED.",
              flush=True)
    debug_vision_dir = ""
    live_rgb_embs: list = []
    live_depth_embs: list = []
    raw_rgb_samples: list = []
    raw_depth_samples: list = []
    RAW_SAMPLE_LIMIT = 10
    if debug_vision:
        debug_vision_dir = os.path.abspath(
            os.path.join(getattr(args_cli, "video_folder", "videos/vision_staircase"), "vision_debug")
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
            depth = camera_data.output["depth"].detach().cpu().numpy()
            depth_emb = encode_depth(depth, defm_model, device)
            if args_cli.disable_rgb:
                # Depth-only, matching collection (--disable_rgb). The dataset has no
                # rgb_embed, so the model's vision input is just the DeFM 1024-dim embed.
                rgb = None
                rgb_emb = None
                vision_embeds = depth_emb.astype(np.float32)
            else:
                rgb = camera_data.output["rgb"].detach().cpu().numpy()
                rgb_emb = encode_rgb(rgb, siglip_model, siglip_processor, device)
                # Concat order is [rgb, depth], matching the training dataset's
                # vision_embeds assembly (g1_offline_dataset.py: rgb then depth).
                vision_embeds = np.concatenate([rgb_emb, depth_emb], axis=0).astype(np.float32)
            if debug_vision and step_count <= 200 and step_count % 20 == 0:
                _debug_vision_step(step_count, rgb, depth, debug_vision_dir)
            if debug_vision:
                if rgb_emb is not None:
                    live_rgb_embs.append(rgb_emb.astype(np.float32))
                live_depth_embs.append(depth_emb.astype(np.float32))
                if len(raw_depth_samples) < RAW_SAMPLE_LIMIT:
                    if rgb is not None:
                        raw_rgb_samples.append(np.asarray(rgb).copy())
                    raw_depth_samples.append(np.asarray(depth).copy())

        if replay_acts is not None:
            # Open-loop replay: ignore obs/vision, feed the recorded action. Hold the
            # last recorded action once we run past the episode length.
            idx = min(step_count, replay_acts.shape[0] - 1)
            action = replay_acts[idx].copy()
        elif guidance_fn is not None:
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
                debug=step_count < args_cli.debug_policy,
            )
        else:
            action = policy.get_action(
                body_pos, body_quat, body_lin_vel, body_ang_vel,
                joint_pos, joint_vel,
                vision_embeds=vision_embeds,
                debug=step_count < args_cli.debug_policy,
            )
        if action is None:
            action = np.zeros(29, dtype=np.float32)

        # Inference-latency compensation (skip in open-loop replay, which feeds
        # recorded actions directly). Apply the action computed `action_shift`
        # steps ago and queue the freshly computed one. The queue is primed with
        # the first prediction so step 0 applies its own action (no spurious
        # zero/default action at start); every later step is delayed by one.
        if args_cli.action_shift > 0 and replay_acts is None:
            action = np.asarray(action, dtype=np.float32)
            if action_queue is None:
                action_queue = deque(
                    [action.copy() for _ in range(args_cli.action_shift)],
                    maxlen=args_cli.action_shift,
                )
            applied_action = action_queue[0].copy()
            action_queue.append(action.copy())  # maxlen evicts the one we just read
            action = applied_action

        # Flushed action diagnostics for the first --debug_policy steps. These are
        # printed by the SCRIPT (not the library) with flush=True so they survive
        # Isaac's hard process teardown, which drops unflushed stdout. `action` here
        # is the unnormalized joint-position target the env will apply: it should be
        # plausible (~order ±1 rad, near the default pose), NOT huge (±5/±10 -> bad
        # model) and NOT stuck in [-1,1] looking normalized (-> normalizer issue).
        if step_count < args_cli.debug_policy:
            a = np.asarray(action, dtype=np.float64)
            robot = env.unwrapped.scene["robot"]
            pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
            print(f"[ACT step {step_count}] action min={a.min():.3f} max={a.max():.3f} "
                  f"mean={a.mean():.3f} absmax={np.abs(a).max():.3f} | pelvis_z={pelvis_z:.3f}m",
                  flush=True)

        action = torch.from_numpy(action).float().to(device).unsqueeze(0)

        if action.shape[0] < env.unwrapped.num_envs:
            action = action.repeat(env.unwrapped.num_envs, 1)
        obs, _, _, _, _ = env.step(action)
        step_count += 1

        if step_count % 100 == 0:
            robot = env.unwrapped.scene["robot"]
            pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
            print(f"Step {step_count}: pelvis height = {pelvis_z:.3f}m", flush=True)

    if keyboard_joystick is not None:
        keyboard_joystick.stop()

    # Gate on depth embeds, not rgb: with --disable_rgb there are no rgb embeds,
    # but we still want the depth embeds/raw depth dumped (this is the whole point
    # of the depth-only OOD analysis). rgb_embed/rgb are saved only if present.
    if debug_vision and live_depth_embs:
        embeds_path = os.path.join(debug_vision_dir, "live_embeds.npz")
        embeds_kwargs = {"depth_embed": np.stack(live_depth_embs, axis=0)}
        if live_rgb_embs:
            embeds_kwargs["rgb_embed"] = np.stack(live_rgb_embs, axis=0)
        np.savez_compressed(embeds_path, **embeds_kwargs)
        print(f"[VISION] Saved {len(live_depth_embs)} live depth embeddings to {embeds_path}", flush=True)
        if raw_depth_samples:
            raw_path = os.path.join(debug_vision_dir, "live_raw_vision.npz")
            raw_kwargs = {"depth": np.stack(raw_depth_samples, axis=0)}
            if raw_rgb_samples:
                raw_kwargs["rgb"] = np.stack(raw_rgb_samples, axis=0)
            np.savez_compressed(raw_path, **raw_kwargs)
            print(f"[VISION] Saved {len(raw_depth_samples)} raw depth samples to {raw_path}",
                  flush=True)

    print(f"[INFO] Completed {step_count} steps", flush=True)
    robot = env.unwrapped.scene["robot"]
    pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
    print(f"[INFO] Final pelvis height = {pelvis_z:.3f}m", flush=True)
    if pelvis_z < 0.3:
        print("[WARNING] Robot appears to have fallen", flush=True)
    else:
        print("[SUCCESS] Robot maintained balance", flush=True)
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
