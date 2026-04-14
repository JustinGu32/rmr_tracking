"""
Kinematic simulation of a KimodoDiT diffusion model in Isaac Lab.

The Kimodo format encodes the world-frame body state as:
    [rp(3) | ra(2) | pelvis_linvel(3) | jp(87) | jv(87) | ja(174) | f(4)]  (360D, full_state_obs=True)
    [rp(3) | ra(2) | jp(87) | jv(87) | ja(174) | f(4)]                     (357D, full_state_obs=False)

where:
  rp             : smoothed pelvis position; xy is RELATIVE to the nominal-frame pelvis XY;
                   z is the absolute world height.
  ra             : [cos(heading), sin(heading)] derived from the hip vector geometry.
  pelvis_linvel  : world-frame pelvis linear velocity (only present in 360D layout).
  jp             : non-root body positions (xy relative to smoothed root, z absolute).
  jv             : world-frame non-root body velocities.
  ja             : world-frame non-root body orientations as 6D rotation features.
  f              : foot contact proxies (4 values).

This script decodes the predicted root state (position, heading, velocity) and writes
it kinematically to Isaac Lab, bypassing physics.  Joint positions are held at the
observed values since the Kimodo format does not encode joint angles directly.

Contrast with kinematic_sim_isaaclab.py which handles G1Dataset (local-frame) models.

Usage:
  python scripts/kinematic_sim_kimodo_isaaclab.py \\
      --task=Tracking-Flat-G1-v0 --checkpoint ckpts/kimodo_model.pt

Headless + save video:
  python scripts/kinematic_sim_kimodo_isaaclab.py \\
      --task=Tracking-Flat-G1-v0 --checkpoint ckpts/kimodo_model.pt \\
      --headless --video --video_folder ./out
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

if os.environ.get("ENABLE_CAMERAS", "") != "1":
    os.environ["ENABLE_CAMERAS"] = "0"

TML_ROOT = Path(__file__).resolve().parent.parent.parent / "TML-BeyondMimic"
RMR_TRACKING_ROOT = Path(__file__).resolve().parent.parent
if RMR_TRACKING_ROOT.exists():
    sys.path.insert(0, str(RMR_TRACKING_ROOT))
    sys.path.insert(0, str(RMR_TRACKING_ROOT / "scripts" / "rsl_rl"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Kinematic sim of KimodoDiT diffusion model in Isaac Lab")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to local checkpoint (.pt)")
parser.add_argument("--wandb_path", type=str, default=None, help="Wandb run path (e.g. user/project/run_id)")
parser.add_argument("--wandb_file", type=str, default="latest.ckpt", help="Checkpoint filename in wandb")
parser.add_argument("--steps", type=int, default=500, help="Number of simulation steps")
parser.add_argument("--deterministic", action="store_true", default=True, help="Deterministic sampling")
parser.add_argument("--task", type=str, default="Tracking-Flat-G1-v0", help="Isaac Lab task")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--motion_file", type=str,
                    default="/move/u/justingu/whole_body_tracking/motions/takara_walk_isaac/motion.npz",
                    help="Path to motion file for tracking command")
parser.add_argument("--video", action="store_true", help="Record simulation to video (offscreen)")
parser.add_argument("--video_folder", type=str, default="videos/kinematic_sim_kimodo",
                    help="Folder to save video")
parser.add_argument("--video_length", type=int, default=500, help="Number of steps to record")
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
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp import time_out as std_time_out

import whole_body_tracking.tasks  # noqa: E402, F401

sys.path.insert(0, str(TML_ROOT))
from diffusion_policy.inference.diffusion_agent import DiffusionAgentIsaac  # noqa: E402
from diffusion_policy.utils.traj_utils import (  # noqa: E402
    quat_from_euler_xyz,
    get_euler_xyz,
)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Kimodo layout constants
# ---------------------------------------------------------------------------

NUM_NONROOT = 29  # BODY_NAMES has 30 entries; pelvis is excluded from body block

# 360D layout (full_state_obs=True):
#   [rp(3) | ra(2) | pelvis_linvel(3) | jp(87) | jv(87) | ja(174) | f(4)]
KIMODO_FULL_RP_SLICE       = slice(0,   3)
KIMODO_FULL_RA_SLICE       = slice(3,   5)
KIMODO_FULL_LINVEL_SLICE   = slice(5,   8)
KIMODO_FULL_JP_SLICE       = slice(8,  95)   # 87 = 29 × 3
KIMODO_FULL_JV_SLICE       = slice(95, 182)  # 87 = 29 × 3
KIMODO_FULL_JA_SLICE       = slice(182, 356) # 174 = 29 × 6
KIMODO_FULL_F_SLICE        = slice(356, 360) # 4
KIMODO_FULL_DIM            = 360

# 357D layout (full_state_obs=False) — no pelvis_linvel:
#   [rp(3) | ra(2) | jp(87) | jv(87) | ja(174) | f(4)]
KIMODO_BASE_RP_SLICE       = slice(0,   3)
KIMODO_BASE_RA_SLICE       = slice(3,   5)
KIMODO_BASE_JP_SLICE       = slice(5,  92)   # 87
KIMODO_BASE_JV_SLICE       = slice(92, 179)  # 87
KIMODO_BASE_JA_SLICE       = slice(179, 353) # 174
KIMODO_BASE_F_SLICE        = slice(353, 357) # 4
KIMODO_BASE_DIM            = 357


# ---------------------------------------------------------------------------
# State decoding
# ---------------------------------------------------------------------------

def _rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Recover a rotation matrix from a 6D representation (Zhou et al. 2019).

    rot6d is the first two *columns* of the rotation matrix, flattened:
        rot6d = [col0(3), col1(3)]  →  shape (N, 6)

    Returns rotation matrices of shape (N, 3, 3).
    """
    N = rot6d.shape[0]
    c0 = rot6d[:, 0:3]
    c1 = rot6d[:, 3:6]
    # Gram-Schmidt orthonormalisation
    c0 = torch.nn.functional.normalize(c0, dim=-1)
    c1 = c1 - (c1 * c0).sum(dim=-1, keepdim=True) * c0
    c1 = torch.nn.functional.normalize(c1, dim=-1)
    c2 = torch.linalg.cross(c0, c1, dim=-1)
    # Stack as columns: (N, 3, 3)
    return torch.stack([c0, c1, c2], dim=-1)  # columns


def _matrix_to_quat(R: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (N, 3, 3) → quaternion (N, 4) in [w, x, y, z] order."""
    N = R.shape[0]
    q = torch.zeros(N, 4, device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt((trace[mask1] + 1.0) * 2.0)  # s = 4*w
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

    # Case 2: R[0,0] largest diagonal
    mask2 = (~mask1) & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt((1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2.0)
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] largest diagonal
    mask3 = (~mask1) & (~mask2) & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt((1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2.0)
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] largest diagonal
    mask4 = (~mask1) & (~mask2) & (~mask3)
    s4 = torch.sqrt((1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2.0)
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4

    return torch.nn.functional.normalize(q, dim=-1)


def decode_kimodo_state(
    state_traj: torch.Tensor,          # (1, horizon, 360 or 357) — NORMALIZED
    next_frame_idx: int,               # which future frame to decode
    current_root_pos_w: torch.Tensor,  # (3,) world-frame pelvis position at nominal frame
    current_root_quat_w: torch.Tensor, # (4,) world-frame pelvis quaternion [w,x,y,z]
    actor,                             # DiffusionActor (for normalizer)
) -> dict:
    """
    Decode the predicted Kimodo state at *next_frame_idx* into world-frame quantities
    suitable for writing to Isaac Lab.

    Kimodo layout:
      360D (full_state_obs=True):
        [rp(3) | ra(2) | pelvis_linvel(3) | jp(87) | jv(87) | ja(174) | f(4)]
      357D (full_state_obs=False):
        [rp(3) | ra(2) | jp(87) | jv(87) | ja(174) | f(4)]

    rp[0:2]  — pelvis XY *relative to the nominal frame* (standardize_xy_to_first_frame).
    rp[2]    — pelvis Z in absolute world coordinates.
    ra       — [cos(heading), sin(heading)]; heading derived from hip geometry (yaw only).
    pelvis_linvel — world-frame pelvis velocity (360D only).

    Returns:
        root_pos_w   (3,) — predicted world-frame pelvis position
        root_quat_w  (4,) — predicted world-frame pelvis quaternion [w,x,y,z]
                            (pitch/roll preserved from current; yaw from ra)
        root_vel_w   (3,) — predicted world-frame pelvis linear velocity (zeros if 357D)
        joint_pos    None — Kimodo does not encode joint angles; caller holds joints at
                            the last observed pose
        joint_vel    None
    """
    obs_dim = state_traj.shape[-1]
    dev = current_root_pos_w.device

    # 1. Unnormalize
    with torch.no_grad():
        state_unnormed = actor.normalizer.unnormalize({"obs": state_traj})["obs"]
    s = state_unnormed[0, next_frame_idx].to(dev)  # (obs_dim,)

    # 2. Select slices based on layout
    if obs_dim == KIMODO_FULL_DIM:
        rp          = s[KIMODO_FULL_RP_SLICE]      # (3,)
        ra          = s[KIMODO_FULL_RA_SLICE]      # (2,)
        pelvis_vel  = s[KIMODO_FULL_LINVEL_SLICE]  # (3,) world-frame
    elif obs_dim == KIMODO_BASE_DIM:
        rp          = s[KIMODO_BASE_RP_SLICE]      # (3,)
        ra          = s[KIMODO_BASE_RA_SLICE]      # (2,)
        pelvis_vel  = torch.zeros(3, device=dev, dtype=torch.float32)
    else:
        print(f"[WARNING] Unexpected Kimodo obs_dim={obs_dim}; expected 360 or 357. "
              "Keeping current root pose.")
        return {
            "root_pos_w":  current_root_pos_w.float(),
            "root_quat_w": current_root_quat_w.float(),
            "root_vel_w":  torch.zeros(3, device=dev, dtype=torch.float32),
            "joint_pos":   None,
            "joint_vel":   None,
        }

    # 3. Root position
    # rp[0:2] is the pelvis XY displacement from the nominal frame (current frame).
    # rp[2]   is the absolute world-height.
    root_pos_w = torch.stack([
        current_root_pos_w[0] + rp[0],
        current_root_pos_w[1] + rp[1],
        rp[2],
    ])

    # 4. Root orientation — preserve observed pitch/roll; update yaw from ra.
    # ra = [cos(heading), sin(heading)] where heading is the robot's forward direction
    # derived from the hip link vector (independent of root pitch/roll).
    roll, pitch, _current_yaw = get_euler_xyz(current_root_quat_w.unsqueeze(0))
    predicted_yaw = torch.atan2(ra[1:2], ra[0:1])  # (1,)
    root_quat_w = quat_from_euler_xyz(roll, pitch, predicted_yaw).squeeze(0)  # (4,)

    return {
        "root_pos_w":  root_pos_w.float(),
        "root_quat_w": root_quat_w.float(),
        "root_vel_w":  pelvis_vel.float(),
        "joint_pos":   None,  # Kimodo does not encode joint angles
        "joint_vel":   None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg):
    print("[INFO] main() entered (Hydra config loaded).", flush=True)

    if args_cli.checkpoint is None and args_cli.wandb_path is None:
        print("[ERROR] No checkpoint specified")
        sys.exit(1)
    if args_cli.checkpoint and args_cli.wandb_path:
        print("[ERROR] Specify only one of --checkpoint or --wandb_path")
        sys.exit(1)

    device = args_cli.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg.seed = seed
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.motion_file:
        env_cfg.commands.motion.motion_file = args_cli.motion_file

    # Extend episode length so the environment doesn't auto-reset mid-run
    record_video = getattr(args_cli, "video", False)
    video_length = getattr(args_cli, "video_length", 500)
    steps_to_seconds = env_cfg.decimation * env_cfg.sim.dt
    episode_s = (max(args_cli.steps, video_length) + 200) * steps_to_seconds
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, episode_s)

    # Disable reference-motion terminations (they fire spuriously for a free-running model)
    if hasattr(env_cfg.terminations, "anchor_pos") and hasattr(env_cfg.terminations.anchor_pos, "params"):
        env_cfg.terminations.anchor_pos.params["threshold"] = 10.0
    if hasattr(env_cfg.terminations, "anchor_ori") and hasattr(env_cfg.terminations.anchor_ori, "params"):
        env_cfg.terminations.anchor_ori.params["threshold"] = 10.0
    if hasattr(env_cfg.terminations, "ee_body_pos") and hasattr(env_cfg.terminations.ee_body_pos, "params"):
        env_cfg.terminations.ee_body_pos.params["threshold"] = 10.0
    if hasattr(env_cfg.terminations, "bad_anchor_pos_xy"):
        env_cfg.terminations.bad_anchor_pos_xy = None
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = DoneTerm(func=std_time_out, time_out=True)
    print(f"[INFO] Relaxed termination thresholds (episode_length_s={env_cfg.episode_length_s:.1f})", flush=True)

    # Load model
    print("[INFO] Loading diffusion policy...", flush=True)
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

    obs_dim = policy.actor.obs_dim
    full_state_obs = getattr(policy.actor, "dataset_full_state_obs", False)
    print(f"[INFO] Model obs_dim={obs_dim}, full_state_obs={full_state_obs}", flush=True)

    if obs_dim == KIMODO_FULL_DIM:
        print("[INFO] 360D Kimodo layout (full_state_obs=True): "
              "root position, heading, and world-frame velocity will be decoded.")
    elif obs_dim == KIMODO_BASE_DIM:
        print("[INFO] 357D Kimodo layout (full_state_obs=False): "
              "root position and heading will be decoded; velocity set to zero.")
    else:
        print(f"[WARNING] obs_dim={obs_dim} does not match known Kimodo layouts "
              f"(expected {KIMODO_FULL_DIM} or {KIMODO_BASE_DIM}). "
              "Decoding may fail.")

    print("[INFO] Note: Kimodo format does not encode joint angles directly. "
          "Joints will be held at the observed position each step.", flush=True)

    # Create environment
    render_mode = "rgb_array" if record_video else None
    print(f"[INFO] Creating environment (render_mode={render_mode!r})...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(
            getattr(args_cli, "video_folder", "videos/kinematic_sim_kimodo")))
        os.makedirs(video_folder, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=video_folder,
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            name_prefix=model_name,
            disable_logger=True,
        )
        print(f"[INFO] Video recording: first {video_length} steps -> {video_folder}")

    policy.reset()
    print("[INFO] Resetting environment...", flush=True)
    obs, _ = env.reset()
    print("[INFO] Environment reset; starting kinematic control loop.", flush=True)

    step_count = 0
    max_steps = args_cli.steps
    env_id = 0
    nom_frame_idx = policy.n_past_steps - 1  # last observed frame = nominal frame

    while step_count < max_steps and simulation_app.is_running():
        # ---- 1. Extract current robot state from observation ----
        dc = obs["diffusion_collect"]
        _idx = env_id if dc["body_pos"].ndim > 1 else slice(None)

        body_pos    = dc["body_pos"][_idx].float().cpu().numpy().reshape(30, 3)
        body_quat   = dc["body_ori"][_idx].float().cpu().numpy().reshape(30, 4)
        body_lin_vel = dc["body_lin_vel"][_idx].float().cpu().numpy().reshape(30, 3)
        body_ang_vel = dc["body_ang_vel"][_idx].float().cpu().numpy().reshape(30, 3)
        joint_pos   = dc["dof_pos"][_idx].float().cpu().numpy()
        joint_vel   = dc["dof_vel"][_idx].float().cpu().numpy()

        # ---- 2. Build obs_dict with history buffer ----
        # DiffusionAgentIsaac.prepare_obs_dict stacks n_past_steps frames and
        # calls G1KimodoMotionDataset.state_normalize internally via actor.act().
        obs_dict = policy.prepare_obs_dict(
            body_pos, body_quat, body_lin_vel, body_ang_vel, joint_pos, joint_vel
        )

        # ---- 3. Run model: get state_traj ----
        past_actions = torch.zeros(
            1, policy.n_past_steps, policy.actor.action_dim,
            device=policy.device, dtype=torch.float32,
        )
        with torch.no_grad():
            action_traj, state_traj = policy.actor.act(
                obs_dict=obs_dict,
                nom_frame_idx=nom_frame_idx,
                past_actions=past_actions,
                use_action_inpainting=False,
            )
        # state_traj: (1, horizon, obs_dim) — normalized Kimodo representation

        # ---- 4. Decode the first predicted future frame ----
        next_frame_idx = policy.n_past_steps
        if next_frame_idx >= state_traj.shape[1]:
            next_frame_idx = state_traj.shape[1] - 1

        robot = env.unwrapped.scene["robot"]
        current_root_pos_w  = robot.data.root_pos_w[env_id].clone()   # (3,)
        current_root_quat_w = robot.data.root_quat_w[env_id].clone()  # (4,) [w,x,y,z]

        decoded = decode_kimodo_state(
            state_traj=state_traj,
            next_frame_idx=next_frame_idx,
            current_root_pos_w=current_root_pos_w,
            current_root_quat_w=current_root_quat_w,
            actor=policy.actor,
        )

        # ---- 5. Write kinematic state to Isaac Lab (bypass physics) ----
        n_envs = env.unwrapped.num_envs
        root_state = torch.zeros(n_envs, 13, device=device)
        root_state[:, 0:3]  = decoded["root_pos_w"].unsqueeze(0).expand(n_envs, -1)
        root_state[:, 3:7]  = decoded["root_quat_w"].unsqueeze(0).expand(n_envs, -1)
        root_state[:, 7:10] = decoded["root_vel_w"].unsqueeze(0).expand(n_envs, -1)
        robot.write_root_state_to_sim(root_state)

        # Kimodo has no joint-angle prediction — hold joints at the last observed pose.
        current_joint_pos = torch.from_numpy(joint_pos).float().to(device).unsqueeze(0)
        current_joint_pos = current_joint_pos.expand(n_envs, -1)
        robot.write_joint_state_to_sim(current_joint_pos, torch.zeros_like(current_joint_pos))
        action_for_step = current_joint_pos

        # ---- 6. Step environment (for rendering + next obs) ----
        if action_for_step.shape[0] < n_envs:
            action_for_step = action_for_step.repeat(n_envs, 1)
        obs, _, _, _, _ = env.step(action_for_step)
        step_count += 1

        if step_count % 100 == 0:
            pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
            print(f"Step {step_count}: pelvis height = {pelvis_z:.3f}m")

    # Summary
    print(f"[INFO] Completed {step_count} kinematic steps")
    robot = env.unwrapped.scene["robot"]
    pelvis_z = robot.data.body_pos_w[env_id, 0, 2].item()
    print(f"[INFO] Final pelvis height = {pelvis_z:.3f}m")
    if pelvis_z < 0.3:
        print("[WARNING] Robot appears to have fallen (or model predicted a low root position)")
    else:
        print("[SUCCESS] Kinematic simulation completed")
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
