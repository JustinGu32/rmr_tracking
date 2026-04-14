"""
Kinematic simulation of a state-prediction diffusion model in Isaac Lab.

For models trained on state-only data (e.g. bones locomotion data where act=0),
the model has learned to predict future body states rather than joint-torque actions.
This script replays those predictions kinematically — directly writing the robot's
predicted root pose (and joint angles, if available) into Isaac Lab at each step,
bypassing the physics engine entirely.

Contrast with sim2sim_isaaclab.py:
  sim2sim:     obs → model → action  → PD controller → physics → new state
  kinematic:   obs → model → state   → write_root_state / write_joint_state (no physics)

The model's actor.act() returns (action_traj, state_traj).  For a bones-trained model
action_traj is meaningless (trained on zeros), but state_traj carries the predicted
body trajectory.  We unnormalize state_traj and decode the next-frame root pose and
(where the body_set includes them) joint positions, then inject them into the sim.

Usage (from rmr_tracking or with PYTHONPATH including rmr_tracking):
  python scripts/kinematic_sim_isaaclab.py --task=Tracking-Flat-G1-v0 --checkpoint ckpts/bones_model.pt

Headless + save video:
  python scripts/kinematic_sim_isaaclab.py --task=Tracking-Flat-G1-v0 \\
      --checkpoint ckpts/bones_model.pt --headless --video --video_folder ./out
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

parser = argparse.ArgumentParser(description="Kinematic sim of bones diffusion model in Isaac Lab")
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
parser.add_argument("--video_folder", type=str, default="videos/kinematic_sim",
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
    quat_rotate,
    box_plus,
)

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# State decoding
# ---------------------------------------------------------------------------

# Joint-encoding block sizes for each joint_repr
_JOINT_DIM = {
    "raw": 29,
    "sincos": 58,
    "rot6d_individual": 174,
    "rot6d_grouped": 78,
}


def decode_next_state(
    state_traj: torch.Tensor,          # (1, horizon, obs_dim) — NORMALIZED
    next_frame_idx: int,               # which frame in the trajectory to decode
    current_root_pos_w: torch.Tensor,  # (3,) world-frame root position now
    current_root_quat_w: torch.Tensor, # (4,) world-frame root quaternion [w,x,y,z]
    actor,                             # DiffusionActor (for normalizer and config)
) -> dict:
    """
    Decode the predicted state at *next_frame_idx* from a normalized state trajectory.

    Handles four state layouts (detected from obs_dim):

      obs_dim == 9  — root_separate=True (root-only diffusion stream):
                      [root_pos_local(3), root_lin_vel_local(3), root_ang_vel_local(3)]

      obs_dim == 192 — full body_set (30 bodies):
                      [body_pos_local(90), body_lin_vel_local(90), root_pos(3),
                       root_rot(3), root_lin_vel(3), root_ang_vel(3)]

      obs_dim == 90  — limited body_set (13 bodies):
                      [body_pos_local(39), body_lin_vel_local(39), root_pos(3),
                       root_rot(3), root_lin_vel(3), root_ang_vel(3)]

      obs_dim >= 58  — reduced / extra_reduced body_set (joint positions):
                      reduced:       [joint_pos(jdim), joint_vel(29), root_pos(3),
                                      root_rot(rrd), root_lin_vel(3), root_ang_vel(3)]
                      extra_reduced: [joint_pos(jdim), joint_vel(29),
                                      root_rot(rrd), root_ang_vel(3)]  (no root_pos)

    Returns a dict with keys:
        root_pos_w   (3,) float32 — predicted world-frame root position
        root_quat_w  (4,) float32 — predicted world-frame root quaternion [w,x,y,z]
        root_vel_w   (3,) float32 — predicted world-frame root linear velocity (may be zeros)
        joint_pos    (29,) float32 or None — predicted joint positions (Isaac ordering)
        joint_vel    (29,) float32 or None — predicted joint velocities (Isaac ordering)
    """
    obs_dim = state_traj.shape[-1]
    dev = current_root_pos_w.device

    # --- 1. Unnormalize state trajectory ---
    with torch.no_grad():
        state_unnormed = actor.normalizer.unnormalize({"obs": state_traj})["obs"]
    next_state = state_unnormed[0, next_frame_idx].to(dev)  # (obs_dim,)

    # --- 2. Compute the nominal-frame yaw quaternion ---
    # This matches _compute_yaw_frame() in g1_offline_dataset.py: strip pitch & roll.
    roll, pitch, yaw = get_euler_xyz(current_root_quat_w.unsqueeze(0))
    yaw_quat = quat_from_euler_xyz(roll * 0.0, pitch * 0.0, yaw).squeeze(0).to(dev)  # (4,)

    # Defaults — overridden below based on layout
    root_pos_w  = current_root_pos_w.clone().to(dev)
    root_quat_w = current_root_quat_w.clone().to(dev)
    root_vel_w  = torch.zeros(3, device=dev, dtype=torch.float32)
    joint_pos_out = None
    joint_vel_out = None

    def _rotate_local_to_world(v_local):
        """Rotate a (3,) local-frame vector to world frame using the nominal yaw."""
        return quat_rotate(yaw_quat.unsqueeze(0), v_local.unsqueeze(0)).squeeze(0)

    def _decode_root_rot(root_rot_local, root_rot_form):
        """Reconstruct world-frame root quaternion from local-frame rotation encoding."""
        if root_rot_form == "original":
            # state stored box_minus(root_rot, yaw_quat); inverse is box_plus
            return box_plus(yaw_quat.unsqueeze(0), root_rot_local.unsqueeze(0)).squeeze(0)
        else:
            # "gravity" or "rot6d": cannot uniquely recover full orientation here.
            # Keep current orientation (yaw is correct; pitch/roll approximated).
            return current_root_quat_w.clone().to(dev)

    # --- 3. Dispatch by obs_dim ---

    if obs_dim == 9:
        # root_separate=True — the diffusion stream only carries root kinematics.
        # Layout: [root_pos_local(3), root_lin_vel_local(3), root_ang_vel_local(3)]
        # There is NO rotation info; keep current root orientation.
        root_pos_local = next_state[0:3]
        root_vel_local = next_state[3:6]
        root_pos_w  = current_root_pos_w.to(dev) + _rotate_local_to_world(root_pos_local)
        root_quat_w = current_root_quat_w.clone().to(dev)  # no rotation in state
        root_vel_w  = _rotate_local_to_world(root_vel_local)

    elif obs_dim == 192:
        # full body_set (30 bodies × 3 = 90D pos, 90D vel, then root block)
        # Layout: [body_pos_local(90), body_lin_vel_local(90),
        #          root_pos(3), root_rot(rrd), root_lin_vel(3), root_ang_vel(3)]
        root_rot_form = getattr(actor, "dataset_root_rot_form", "original")
        rrd = 6 if root_rot_form == "rot6d" else 3
        root_pos_local  = next_state[180:183]
        root_rot_local  = next_state[183: 183 + rrd]
        root_vel_local  = next_state[183 + rrd: 183 + rrd + 3]
        root_pos_w  = current_root_pos_w.to(dev) + _rotate_local_to_world(root_pos_local)
        root_quat_w = _decode_root_rot(root_rot_local, root_rot_form)
        if root_vel_local.numel() == 3:
            root_vel_w = _rotate_local_to_world(root_vel_local)

    elif obs_dim == 90:
        # limited body_set (13 bodies × 3 = 39D pos, 39D vel, then root block)
        # Layout: [body_pos_local(39), body_lin_vel_local(39),
        #          root_pos(3), root_rot(rrd), root_lin_vel(3), root_ang_vel(3)]
        root_rot_form = getattr(actor, "dataset_root_rot_form", "original")
        rrd = 6 if root_rot_form == "rot6d" else 3
        root_pos_local  = next_state[78:81]
        root_rot_local  = next_state[81: 81 + rrd]
        root_vel_local  = next_state[81 + rrd: 81 + rrd + 3]
        root_pos_w  = current_root_pos_w.to(dev) + _rotate_local_to_world(root_pos_local)
        root_quat_w = _decode_root_rot(root_rot_local, root_rot_form)
        if root_vel_local.numel() == 3:
            root_vel_w = _rotate_local_to_world(root_vel_local)

    elif obs_dim >= 58:
        # reduced or extra_reduced body_set.  Infer params from actor attributes.
        root_rot_form = getattr(actor, "dataset_root_rot_form", "original")
        joint_repr    = getattr(actor, "dataset_joint_repr", "raw") or "raw"
        body_set      = getattr(actor, "dataset_body_set", None) or "reduced"
        jdim = _JOINT_DIM.get(joint_repr, 29)
        rrd  = 6 if root_rot_form == "rot6d" else 3

        # Decode joint positions (raw only; other reprs require inversion not done here)
        if joint_repr == "raw" and obs_dim >= jdim * 2:
            joint_pos_out = next_state[0:29].float()
            joint_vel_out = next_state[29:58].float()

        if body_set == "extra_reduced":
            # No root_pos in extra_reduced — root translation cannot be decoded.
            # Layout: [joint_pos(jdim), joint_vel(29), root_rot(rrd), root_ang_vel(3)]
            rr_start = jdim + 29
            root_rot_local = next_state[rr_start: rr_start + rrd]
            root_quat_w = _decode_root_rot(root_rot_local, root_rot_form)
            # root_pos_w stays as current_root_pos_w (no update)
        else:
            # reduced — has root_pos.
            # Layout: [joint_pos(jdim), joint_vel(29), root_pos(3),
            #          root_rot(rrd), root_lin_vel(3), root_ang_vel(3)]
            rp_start = jdim + 29
            rr_start = rp_start + 3
            rv_start = rr_start + rrd
            root_pos_local = next_state[rp_start: rp_start + 3]
            root_rot_local = next_state[rr_start: rr_start + rrd]
            root_pos_w  = current_root_pos_w.to(dev) + _rotate_local_to_world(root_pos_local)
            root_quat_w = _decode_root_rot(root_rot_local, root_rot_form)
            root_vel_local = next_state[rv_start: rv_start + 3]
            if root_vel_local.numel() == 3:
                root_vel_w = _rotate_local_to_world(root_vel_local)

    else:
        print(f"[WARNING] Unknown obs_dim={obs_dim}; cannot decode state. "
              "Keeping current root pose.")

    return {
        "root_pos_w": root_pos_w.float(),
        "root_quat_w": root_quat_w.float(),
        "root_vel_w": root_vel_w.float(),
        "joint_pos": joint_pos_out,
        "joint_vel": joint_vel_out,
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
    body_set = getattr(policy.actor, "dataset_body_set", None) or "limited"
    root_rot_form = getattr(policy.actor, "dataset_root_rot_form", "original")
    joint_repr = getattr(policy.actor, "dataset_joint_repr", "raw")
    print(f"[INFO] Model obs_dim={obs_dim}, body_set={body_set!r}, "
          f"root_rot_form={root_rot_form!r}, joint_repr={joint_repr!r}", flush=True)

    if obs_dim == 9:
        print("[INFO] root_separate=True detected (obs_dim=9): model predicts root trajectory only. "
              "Root position will be set kinematically; orientation and joints held at observed pose.")
    elif body_set == "reduced" and joint_repr == "raw":
        print("[INFO] Model uses reduced body_set with raw joints — joint positions will be decoded from state.")
    else:
        print("[INFO] Root pose will be set kinematically; "
              "joints will be held at the observed pose (no joint prediction).")

    # Create environment
    render_mode = "rgb_array" if record_video else None
    print(f"[INFO] Creating environment (render_mode={render_mode!r})...", flush=True)
    env = gym.make(args_cli.task, cfg=env_cfg, device=device, render_mode=render_mode, seed=seed)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    if record_video:
        video_folder = os.path.abspath(os.path.expanduser(getattr(args_cli, "video_folder", "videos/kinematic_sim")))
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

    # Pre-allocate zero actions (used only to drive the env.step call for rendering)
    zero_action = torch.zeros(1, policy.actor.action_dim, device=device)

    while step_count < max_steps and simulation_app.is_running():
        # ---- 1. Extract current robot state from observation ----
        dc = obs["diffusion_collect"]
        _idx = env_id if dc["body_pos"].ndim > 1 else slice(None)

        body_pos   = dc["body_pos"][_idx].float().cpu().numpy().reshape(30, 3)
        body_quat  = dc["body_ori"][_idx].float().cpu().numpy().reshape(30, 4)
        body_lin_vel = dc["body_lin_vel"][_idx].float().cpu().numpy().reshape(30, 3)
        body_ang_vel = dc["body_ang_vel"][_idx].float().cpu().numpy().reshape(30, 3)
        joint_pos  = dc["dof_pos"][_idx].float().cpu().numpy()
        joint_vel  = dc["dof_vel"][_idx].float().cpu().numpy()

        # ---- 2. Build obs_dict with history buffer (same as sim2sim) ----
        obs_dict = policy.prepare_obs_dict(
            body_pos, body_quat, body_lin_vel, body_ang_vel, joint_pos, joint_vel
        )

        # ---- 3. Run model: get state_traj (and action_traj — ignored for bones model) ----
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
        # state_traj: (1, horizon, obs_dim) — normalized

        # ---- 4. Decode predicted next state from state_traj ----
        # The first FUTURE frame is at index n_past_steps.
        next_frame_idx = policy.n_past_steps
        if next_frame_idx >= state_traj.shape[1]:
            next_frame_idx = state_traj.shape[1] - 1

        robot = env.unwrapped.scene["robot"]
        current_root_pos_w  = robot.data.root_pos_w[env_id].clone()   # (3,)
        current_root_quat_w = robot.data.root_quat_w[env_id].clone()  # (4,) [w,x,y,z]

        decoded = decode_next_state(
            state_traj=state_traj,
            next_frame_idx=next_frame_idx,
            current_root_pos_w=current_root_pos_w,
            current_root_quat_w=current_root_quat_w,
            actor=policy.actor,
        )

        # ---- 5. Write kinematic state to Isaac Lab (bypass physics) ----
        #
        # root_state: (n_envs, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        # We set position + orientation + estimated velocity; angular velocity = 0.
        n_envs = env.unwrapped.num_envs
        root_state = torch.zeros(n_envs, 13, device=device)
        root_state[:, 0:3] = decoded["root_pos_w"].unsqueeze(0).expand(n_envs, -1)
        root_state[:, 3:7] = decoded["root_quat_w"].unsqueeze(0).expand(n_envs, -1)
        root_state[:, 7:10] = decoded["root_vel_w"].unsqueeze(0).expand(n_envs, -1)
        robot.write_root_state_to_sim(root_state)

        if decoded["joint_pos"] is not None:
            # Predicted joint positions available (reduced body_set) — write them
            jp = decoded["joint_pos"].unsqueeze(0).expand(n_envs, -1)  # (n_envs, 29)
            jv = decoded["joint_vel"].unsqueeze(0).expand(n_envs, -1)  # (n_envs, 29)
            robot.write_joint_state_to_sim(jp, jv)
            # Also pass predicted joint positions as action so the PD controller
            # holds the joints at the desired angles during the physics substep.
            action_for_step = jp
        else:
            # No joint prediction — hold joints at their current observed positions
            # so the PD controller keeps them there during the physics substep.
            current_joint_pos = torch.from_numpy(joint_pos).float().to(device).unsqueeze(0)
            current_joint_pos = current_joint_pos.expand(n_envs, -1)
            robot.write_joint_state_to_sim(current_joint_pos,
                                            torch.zeros_like(current_joint_pos))
            action_for_step = current_joint_pos

        # ---- 6. Step environment (advances sim for rendering + gives next obs) ----
        # The physics runs one decimation cycle from the kinematic state we just wrote.
        # We override the state again at the next iteration, so any physics drift is
        # corrected every step.
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
