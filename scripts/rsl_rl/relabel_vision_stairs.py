"""Script to relabel vision data from a previously collected zarr dataset
by replaying kinematic states in an Isaac Lab scene with a staircase,
re-rendering camera views, and recomputing vision embeddings.

Saves both original and horizontally-flipped embeddings for each frame to
support symmetry-based data augmentation at training time
(see https://arxiv.org/abs/2403.04359).

Example command:
    python scripts/rsl_rl/relabel_vision_stairs.py \
        --task Tracking-Flat-G1-Collect-v0 \
        --enable_cameras \
        --headless \
        --old_zarr_path /path/to/old_dataset.zarr \
        --new_zarr_path relabeled_staircase_dataset.zarr \
        --staircase_ahead_m 3.0 \
        --cutoff_margin 0.5
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import warnings

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Relabel vision data with staircase in scene.")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-Collect-v0", help="Name of the task.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--old_zarr_path", type=str,
    default="/move/u/justingu/rmr_tracking/VISION_DATA_COLLECTED/2026-01-28_17-39-47_takara_walk_isaac_npz_ep-10000_steps-60_delay-0-0_noise-0.3_hip-0.3_knee-0.3_ankle-0.5_03_2021.zarr",
    help="Path to the original zarr dataset.")
parser.add_argument("--new_zarr_path", type=str, default="relabeled_staircase_dataset.zarr",
    help="Path for the output relabeled zarr dataset.")
parser.add_argument("--with_stairs", dest="with_stairs", action="store_true",
    help="Replay trajectories with the staircase inserted into the scene.")
parser.add_argument("--without_stairs", dest="with_stairs", action="store_false",
    help="Replay trajectories without inserting a staircase, useful for creating a matched no-stairs zarr.")
parser.set_defaults(with_stairs=True)
parser.add_argument("--staircase_ahead_m", type=float, default=0.15,
    help="Place the first step face this many metres ahead of each trajectory end, measured along the episode's final facing direction.")
parser.add_argument("--staircase_x", type=float, default=None,
    help=argparse.SUPPRESS)
parser.add_argument("--staircase_yaw_bias_deg", type=float, default=0,
    help="Extra yaw bias in degrees, applied per episode to show the staircase at a slight angle in the camera.")
parser.add_argument("--cutoff_margin", type=float, default=0.35,
    help="Stop collecting when the robot root is within this distance of the first step face.")
parser.add_argument("--max_episodes", type=int, default=None,
    help="Max number of episodes to relabel (default: all).")
parser.add_argument("--save_video", action="store_true", default=False,
    help="Save per-episode MP4 videos of the camera view.")
parser.add_argument("--max_video_episodes", type=int, default=10,
    help="Max number of episodes to save as video (to limit disk usage).")
parser.add_argument("--video_fps", type=float, default=30.0,
    help="Playback FPS for saved MP4s. Around 30-33 fps matches 60 collected steps to about 2 seconds of motion.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras for vision relabeling
args_cli.enable_cameras = True

# Disable obstacle spawning — we only want the staircase
os.environ["NUM_OBSTACLES"] = "0"

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
import zarr
import numpy as np
import imageio
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.assets import RigidObjectCfg

# Set ENABLE_CAMERAS before importing tasks so config files pick it up
os.environ["ENABLE_CAMERAS"] = "1"
import whole_body_tracking.tasks  # registers gym envs  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

# Vision models
from transformers import AutoImageProcessor, SiglipModel
from whole_body_tracking.utils.defm_utils import preprocess_depth_batch

# Replay buffer for loading and saving
from replay_buffer import ReplayBuffer


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    """Return world yaw from a quaternion stored as (w, x, y, z)."""
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(np.arctan2(siny_cosp, cosy_cosp))


def get_first_step_face_offset_m() -> float:
    """Distance from staircase root to the lowest step face along ascent."""
    box1_path = Path("/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase/box_models/box1.obj")
    staircase_scale = 0.8380952380952381
    default_offset_m = 0.345 * staircase_scale

    try:
        min_y = None
        with open(box1_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    y = float(line.split()[2])
                    min_y = y if min_y is None else min(min_y, y)
        if min_y is None:
            raise ValueError("No vertices found in staircase mesh.")
        return abs(min_y) * staircase_scale
    except (OSError, ValueError, IndexError):
        warnings.warn(
            f"Falling back to hard-coded first-step offset {default_offset_m:.4f} m",
            stacklevel=2,
        )
        return default_offset_m


def load_stair_boxes_local():
    """Load staircase box extents in the staircase local frame."""
    mesh_dir = Path("/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase/box_models")
    staircase_scale = 0.8380952380952381
    box_names = ["box1.obj", "box2.obj", "box3.obj"]
    boxes = []

    for name in box_names:
        verts = []
        with open(mesh_dir / name, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("v "):
                    _, x, y, z = line.split()[:4]
                    verts.append([float(x), float(y), float(z)])
        verts = np.asarray(verts, dtype=np.float32) * staircase_scale
        boxes.append((verts.min(axis=0), verts.max(axis=0)))
    return boxes


def body_collides_with_staircase(
    body_pos_world: np.ndarray,
    staircase_pos_world_xy: np.ndarray,
    staircase_yaw: float,
    stair_boxes_local,
    xy_margin: float = 0.015,
    z_margin: float = 0.01,
) -> bool:
    """Detect whether any tracked body point enters one of the stair volumes."""
    rel_xy = body_pos_world[:, :2] - staircase_pos_world_xy[None, :]
    c = float(np.cos(staircase_yaw))
    s = float(np.sin(staircase_yaw))
    local_x = c * rel_xy[:, 0] + s * rel_xy[:, 1]
    local_y = -s * rel_xy[:, 0] + c * rel_xy[:, 1]
    local_z = body_pos_world[:, 2]

    for box_min, box_max in stair_boxes_local:
        inside_x = (local_x >= box_min[0] - xy_margin) & (local_x <= box_max[0] + xy_margin)
        inside_y = (local_y >= box_min[1] - xy_margin) & (local_y <= box_max[1] + xy_margin)
        inside_z = (local_z >= box_min[2] - z_margin) & (local_z <= box_max[2] + z_margin)
        if np.any(inside_x & inside_y & inside_z):
            return True
    return False


def encode_rgb_pair(rgb_np: np.ndarray, processor, siglip_model, device):
    """Encode original and horizontally-flipped RGB image in one batched forward pass.

    Args:
        rgb_np: (H, W, 3) uint8 array.
    Returns:
        (orig_embed, flip_embed): two (1, embed_dim) numpy arrays.
    """
    rgb_np_flip = rgb_np[:, ::-1, :].copy()  # left-right mirror
    images = [Image.fromarray(rgb_np), Image.fromarray(rgb_np_flip)]
    inputs = processor(images=images, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.float16)
    outputs = siglip_model.vision_model(**inputs)
    pooled = outputs.pooler_output.cpu().numpy()  # (2, embed_dim)
    return pooled[0:1], pooled[1:2], rgb_np_flip


def encode_depth_pair(depth_np: np.ndarray, defm_model, device):
    """Encode original and horizontally-flipped depth image in one batched forward pass.

    Args:
        depth_np: (H, W) float array.
    Returns:
        (orig_embed, flip_embed): two (1, 1024) numpy arrays.
    """
    depth_np_flip = depth_np[:, ::-1].copy()  # left-right mirror
    batch = np.stack([depth_np, depth_np_flip], axis=0)  # (2, H, W)
    norm_depth = preprocess_depth_batch(
        batch, target_size=518, patch_size=14, device=device,
    ).float()
    out = defm_model.get_intermediate_layers(
        norm_depth, n=1, reshape=True, return_class_token=True,
    )
    class_tokens = out[0][1].cpu().numpy()  # (2, 1024)
    return class_tokens[0:1], class_tokens[1:2]


def main():
    device = args_cli.device if hasattr(args_cli, 'device') and args_cli.device else ("cuda" if torch.cuda.is_available() else "cpu")

    if args_cli.staircase_x is not None:
        warnings.warn(
            "--staircase_x is deprecated; use --staircase_ahead_m instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        args_cli.staircase_ahead_m = args_cli.staircase_x

    STAIRCASE_AHEAD_M = args_cli.staircase_ahead_m
    STAIRCASE_YAW_BIAS_RAD = np.deg2rad(args_cli.staircase_yaw_bias_deg)
    DISTANCE_THRESHOLD = args_cli.cutoff_margin
    FIRST_STEP_FACE_OFFSET_M = get_first_step_face_offset_m() if args_cli.with_stairs else 0.0
    STAIR_BOXES_LOCAL = load_stair_boxes_local() if args_cli.with_stairs else None

    # ---------------------------------------------------------
    # 1. LOAD PREVIOUS DATASET via ReplayBuffer
    # ---------------------------------------------------------
    print(f"[INFO] Loading original dataset: {args_cli.old_zarr_path}")
    old_buff = ReplayBuffer.create_from_path(args_cli.old_zarr_path, mode='r')
    num_episodes = old_buff.n_episodes
    print(f"[INFO] Found {num_episodes} episodes, {old_buff.n_steps} total steps")

    if args_cli.max_episodes is not None:
        num_episodes = min(num_episodes, args_cli.max_episodes)
        print(f"[INFO] Limiting to {num_episodes} episodes")

    # ---------------------------------------------------------
    # 2. SETUP ISAAC SCENE (NO RL WRAPPERS)
    # ---------------------------------------------------------
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=1,
        use_fabric=not args_cli.disable_fabric,
    )

    scene_cfg = env_cfg.scene
    scene_cfg.num_envs = 1
    if scene_cfg.depth_camera is not None:
        scene_cfg.depth_camera.update_period = env_cfg.sim.dt

    if args_cli.with_stairs:
        scene_cfg.staircase = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Staircase",
            spawn=sim_utils.UrdfFileCfg(
                asset_path="/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase/multi_boxes_scaled_0.84_0.84_0.84.urdf",
                usd_dir=os.path.expanduser("~/tmp/IsaacLab/staircase_usd"),
                fix_base=True,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.0),
                rot=(0.7071068, 0.0, 0.0, -0.7071068),  # (w, x, y, z)
            ),
        )

    sim_cfg = env_cfg.sim
    sim_cfg.device = device
    sim = sim_utils.SimulationContext(sim_cfg)
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    sim_dt = sim.get_physics_dt()
    robot = scene["robot"]
    camera = scene["depth_camera"]
    env_origin = scene.env_origins[0]
    print(f"[INFO] env_origin = {env_origin}")

    # ---------------------------------------------------------
    # 3. INITIALIZE VISION MODELS
    # ---------------------------------------------------------
    print("[INFO] Initializing Siglip2 Vision Model...")
    model_id = "google/siglip2-so400m-patch14-384"
    processor = AutoImageProcessor.from_pretrained(model_id)
    try:
        siglip_model = SiglipModel.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            dtype=torch.float16,
        ).to(device=device).eval()
    except (ImportError, ValueError):
        print("[WARN] flash_attention_2 not available, using default attention")
        siglip_model = SiglipModel.from_pretrained(
            model_id,
            dtype=torch.float16,
        ).to(device=device).eval()
    rgb_embed_dim = siglip_model.vision_model.config.hidden_size

    print("[INFO] Initializing DeFM Depth Model...")
    torch.hub.set_dir(os.path.expanduser("~/.cache/torch/hub"))
    defm_model = torch.hub.load(
        "leggedrobotics/defm:main",
        "defm_vit_l14",
        pretrained=True,
        trust_repo=True,
    ).eval().to(device=device)
    depth_embed_dim = 1024

    # ---------------------------------------------------------
    # 4. INITIALIZE NEW REPLAY BUFFER
    # ---------------------------------------------------------
    print(f"[INFO] Output will be saved to: {args_cli.new_zarr_path}")
    new_buff = ReplayBuffer.create_empty_zarr(
        storage=zarr.DirectoryStore(args_cli.new_zarr_path)
    )

    metadata = {
        'source_zarr': str(args_cli.old_zarr_path),
        'with_stairs': args_cli.with_stairs,
        'staircase_ahead_m': STAIRCASE_AHEAD_M,
        'staircase_distance': STAIRCASE_AHEAD_M,
        'staircase_yaw_bias_deg': args_cli.staircase_yaw_bias_deg,
        'first_step_face_offset_m': FIRST_STEP_FACE_OFFSET_M,
        'cutoff_margin': DISTANCE_THRESHOLD,
        'relabeled': True,
        'has_flip_embeds': True,  # NEW: signal to downstream consumers
    }
    new_buff.update_meta(metadata)

    # ---------------------------------------------------------
    # 5. KINEMATIC REPLAY & RELABEL LOOP
    # ---------------------------------------------------------
    print("[INFO] Starting Relabeling Process...")
    saved_episodes = 0
    skipped_episodes = 0
    collision_episodes = 0

    debug_dir = Path(args_cli.new_zarr_path).parent / "debug_frames"
    debug_dir.mkdir(parents=True, exist_ok=True)

    if args_cli.save_video:
        video_dir = Path(args_cli.new_zarr_path).parent / "videos"
        video_dir.mkdir(parents=True, exist_ok=True)
        video_fps = args_cli.video_fps
        print(f"[INFO] Video recording enabled: saving to {video_dir} at {video_fps} fps")

    for ep in tqdm(range(num_episodes), desc="Relabeling episodes"):
        ep_data = old_buff.get_episode(ep)

        ep_root_pos = torch.tensor(ep_data['root_pos'], device=device, dtype=torch.float32)
        ep_root_rot = torch.tensor(ep_data['root_rot'], device=device, dtype=torch.float32)
        ep_joint_pos = torch.tensor(ep_data['joint_pos'], device=device, dtype=torch.float32)
        ep_joint_vel = torch.tensor(ep_data['joint_vel'], device=device, dtype=torch.float32)

        ep_steps = ep_root_pos.shape[0]

        # --- PER-EPISODE STAIRCASE PLACEMENT ---
        end_pos_xy = ep_root_pos[-1, :2].cpu().numpy()
        final_root_quat = ep_root_rot[-1].cpu().numpy()
        facing_yaw = yaw_from_quat_wxyz(final_root_quat)
        facing_dir = np.array([np.cos(facing_yaw), np.sin(facing_yaw)], dtype=np.float32)

        k = min(10, ep_steps - 1)
        start_window = ep_root_pos[-(k+1), :2].cpu().numpy()
        walk_dir = end_pos_xy - start_window
        walk_dist = np.linalg.norm(walk_dir)
        if walk_dist > 1e-3:
            walk_dir = walk_dir / walk_dist
        else:
            walk_dir = end_pos_xy - ep_root_pos[0, :2].cpu().numpy()
            d = np.linalg.norm(walk_dir)
            walk_dir = walk_dir / d if d > 1e-3 else np.array([1.0, 0.0])

        facing_norm = np.linalg.norm(facing_dir)
        if not np.isfinite(facing_norm) or facing_norm < 1e-6:
            facing_dir = walk_dir.astype(np.float32)
            facing_yaw = float(np.arctan2(facing_dir[1], facing_dir[0]))
        else:
            facing_dir = facing_dir / facing_norm

        if args_cli.with_stairs:
            first_step_xy = end_pos_xy + facing_dir * STAIRCASE_AHEAD_M
            staircase_xy = first_step_xy + facing_dir * FIRST_STEP_FACE_OFFSET_M
            staircase_pos_world = torch.tensor(
                [staircase_xy[0] + env_origin[0].item(),
                 staircase_xy[1] + env_origin[1].item(),
                 0.0 + env_origin[2].item()],
                device=device, dtype=torch.float32,
            )

            total_yaw = facing_yaw - np.pi / 2.0 + STAIRCASE_YAW_BIAS_RAD
            half = total_yaw / 2.0
            stair_quat = torch.tensor(
                [np.cos(half), 0.0, 0.0, np.sin(half)],
                device=device, dtype=torch.float32,
            )

            staircase_asset = scene["staircase"]
            stair_state = staircase_asset.data.root_state_w[0:1].clone()
            stair_state[0, 0:3] = staircase_pos_world
            stair_state[0, 3:7] = stair_quat
            stair_state[0, 7:] = 0.0
            staircase_asset.write_root_state_to_sim(stair_state, env_ids=torch.tensor([0], device=device))
        else:
            first_step_xy = None
            staircase_xy = None
            total_yaw = facing_yaw

        if ep < 5:
            traj_yaw = float(np.arctan2(walk_dir[1], walk_dir[0]))
            if args_cli.with_stairs:
                print(f"  [ep {ep}] traj: ({ep_root_pos[0,0].item():.1f},{ep_root_pos[0,1].item():.1f}) → "
                      f"({end_pos_xy[0]:.1f},{end_pos_xy[1]:.1f}), "
                      f"first_step at ({first_step_xy[0]:.1f},{first_step_xy[1]:.1f}), "
                      f"stair_root at ({staircase_xy[0]:.1f},{staircase_xy[1]:.1f}), "
                      f"traj_yaw={np.degrees(traj_yaw):.0f}°, "
                      f"facing_yaw={np.degrees(facing_yaw):.0f}°, "
                      f"total_yaw={np.degrees(total_yaw):.0f}°")
            else:
                print(f"  [ep {ep}] traj: ({ep_root_pos[0,0].item():.1f},{ep_root_pos[0,1].item():.1f}) → "
                      f"({end_pos_xy[0]:.1f},{end_pos_xy[1]:.1f}), "
                      f"traj_yaw={np.degrees(traj_yaw):.0f}°, "
                      f"facing_yaw={np.degrees(facing_yaw):.0f}°, no stairs")

        # Per-episode embedding accumulators (originals + flips)
        ep_rgb_embeds = []
        ep_rgb_embeds_flip = []
        ep_depth_embeds = []
        ep_depth_embeds_flip = []
        ep_video_frames = []
        valid_steps = 0
        collision_detected = False
        collision_step = None

        for step in range(ep_steps):
          with torch.inference_mode():
            if args_cli.with_stairs:
                pos_xy = ep_root_pos[step, :2].cpu().numpy()
                dist_to_first_step = float(np.linalg.norm(pos_xy - first_step_xy))
                if dist_to_first_step < DISTANCE_THRESHOLD:
                    break

                body_pos_step = np.asarray(ep_data["body_pos"][step], dtype=np.float32).reshape(-1, 3)
                if body_collides_with_staircase(
                    body_pos_world=body_pos_step,
                    staircase_pos_world_xy=staircase_xy,
                    staircase_yaw=total_yaw,
                    stair_boxes_local=STAIR_BOXES_LOCAL,
                ):
                    collision_detected = True
                    collision_step = step
                    break

            # --- OVERRIDE SIMULATOR STATE ---
            root_state = torch.zeros(1, 13, device=device)
            root_state[0, 0:3] = ep_root_pos[step] + env_origin
            root_state[0, 3:7] = ep_root_rot[step]
            if 'body_lin_vel' in ep_data:
                num_bodies = ep_data['body_lin_vel'].shape[-1] // 3
                root_lin_vel = ep_data['body_lin_vel'][step].reshape(num_bodies, 3)[0]
                root_state[0, 7:10] = torch.tensor(root_lin_vel, device=device, dtype=torch.float32)
            if 'body_ang_vel' in ep_data:
                num_bodies = ep_data['body_ang_vel'].shape[-1] // 3
                root_ang_vel = ep_data['body_ang_vel'][step].reshape(num_bodies, 3)[0]
                root_state[0, 10:13] = torch.tensor(root_ang_vel, device=device, dtype=torch.float32)

            robot.write_root_state_to_sim(root_state, env_ids=torch.tensor([0], device=device))
            robot.write_joint_state_to_sim(
                ep_joint_pos[step].unsqueeze(0),
                ep_joint_vel[step].unsqueeze(0),
                env_ids=torch.tensor([0], device=device),
            )

            sim.step(render=True)
            scene.update(dt=sim.get_physics_dt())

            # --- EXTRACT IMAGES ---
            rgb_image = camera.data.output["rgb"][0]
            depth_image = camera.data.output["depth"][0]

            rgb_np = rgb_image[..., :3].cpu().numpy().astype(np.uint8)
            depth_np = depth_image.squeeze(-1).cpu().numpy()

            # --- ENCODE RGB (original + horizontal flip in one batched pass) ---
            rgb_orig_emb, rgb_flip_emb, rgb_np_flip = encode_rgb_pair(
                rgb_np, processor, siglip_model, device,
            )
            ep_rgb_embeds.append(rgb_orig_emb)
            ep_rgb_embeds_flip.append(rgb_flip_emb)

            # --- ENCODE DEPTH (original + horizontal flip in one batched pass) ---
            depth_orig_emb, depth_flip_emb = encode_depth_pair(depth_np, defm_model, device)
            ep_depth_embeds.append(depth_orig_emb)
            ep_depth_embeds_flip.append(depth_flip_emb)

            # Save debug frames for first episode (both original and flipped)
            if ep == 0 and step % 10 == 0:
                Image.fromarray(rgb_np).save(str(debug_dir / f"ep0_step{step:04d}.png"))
                Image.fromarray(rgb_np_flip).save(str(debug_dir / f"ep0_step{step:04d}_flip.png"))

            # Accumulate frames for video (only originals; videos are sanity checks)
            if args_cli.save_video and saved_episodes < args_cli.max_video_episodes:
                ep_video_frames.append(rgb_np.copy())

            valid_steps += 1

            if step % 10 == 0:
                if args_cli.with_stairs:
                    print(f"  ep {ep} step {step}/{ep_steps}  dist_to_first_step={dist_to_first_step:.2f}m")
                else:
                    print(f"  ep {ep} step {step}/{ep_steps}")

        # --- SAVE EPISODE ---
        if collision_detected:
            skipped_episodes += 1
            collision_episodes += 1
            print(f"[INFO] Skipping ep {ep}: staircase collision detected at step {collision_step}.")
        elif valid_steps > 0:
            episode_data = {
                "body_pos": np.array(ep_data['body_pos'][:valid_steps]),
                "body_rot": np.array(ep_data['body_rot'][:valid_steps]),
                "joint_pos": np.array(ep_data['joint_pos'][:valid_steps]),
                "joint_vel": np.array(ep_data['joint_vel'][:valid_steps]),
                "root_pos": np.array(ep_data['root_pos'][:valid_steps]),
                "root_rot": np.array(ep_data['root_rot'][:valid_steps]),
                "act": np.array(ep_data['act'][:valid_steps]),
                "rgb_embed":           np.concatenate(ep_rgb_embeds, axis=0),
                "rgb_embed_flipped":   np.concatenate(ep_rgb_embeds_flip, axis=0),
                "depth_embed":         np.concatenate(ep_depth_embeds, axis=0),
                "depth_embed_flipped": np.concatenate(ep_depth_embeds_flip, axis=0),
            }

            if 'body_lin_vel' in ep_data:
                episode_data["body_lin_vel"] = np.array(ep_data['body_lin_vel'][:valid_steps])
            if 'body_ang_vel' in ep_data:
                episode_data["body_ang_vel"] = np.array(ep_data['body_ang_vel'][:valid_steps])

            new_buff.add_episode(episode_data)
            saved_episodes += 1

            if args_cli.save_video and len(ep_video_frames) > 0 and saved_episodes <= args_cli.max_video_episodes:
                vid_path = str(video_dir / f"ep{ep:04d}.mp4")
                imageio.mimwrite(vid_path, ep_video_frames, fps=video_fps, quality=8)
                print(f"[INFO] Saved video ({len(ep_video_frames)} frames) → {vid_path}")

            if saved_episodes % 100 == 0:
                print(f"[INFO] Saved {saved_episodes} episodes so far...")
        else:
            skipped_episodes += 1

    print(f"\n[INFO] Relabeling Complete!")
    print(f"  Saved episodes: {saved_episodes}")
    print(f"  Skipped episodes (total): {skipped_episodes}")
    print(f"  Skipped episodes (stair collisions): {collision_episodes}")
    print(f"  Output: {args_cli.new_zarr_path}")

    simulation_app.close()


if __name__ == "__main__":
    main()
