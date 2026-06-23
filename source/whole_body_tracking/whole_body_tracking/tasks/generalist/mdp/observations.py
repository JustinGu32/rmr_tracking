from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv, quat_mul

from whole_body_tracking.tasks.generalist.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    grav_dir = quat_apply(quat_inv(command.robot_anchor_quat_w), command.down_dir)
    return grav_dir.view(env.num_envs, -1)


def vr_3point_local_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_compliant_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ext_force_disp_w = command.last_force_applied * command.eef_stiffness_buf[:,:,None]
    root_quat = command.robot_anchor_quat_w[:,None,:].repeat(1, len(command.cfg.vr_3point_body),1)
    ext_force_disp_l = quat_apply(quat_inv(root_quat), ext_force_disp_w)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:,None,:]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    ref_3point_root -= ext_force_disp_l
    return ref_3point_root.view(env.num_envs, -1)


def vr_3point_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_quat = command.vr_3point_body_quat_w
    ref_3point_root = quat_mul(quat_inv(ref_root_quat), ref_3point_quat)
    return ref_3point_root.view(env.num_envs, -1)


def last_action_pseudotarget(env: ManagerBasedEnv) -> torch.Tensor:
    """Last action in pseudo-target units: (processed_actions - default_pos) / scale.

    For target mode this equals the raw PPO output (backward compatible).
    For delta mode this equals (x_ref + delta - default_pos) / scale, matching the ONNX output format.
    """
    action_term = env.action_manager.get_term("joint_pos")
    default_pos = env.scene["robot"].data.default_joint_pos
    return (action_term.processed_actions - default_pos) / action_term._scale


def compliance(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.eef_stiffness_buf * 10.0


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

# For Diffusion

def robot_root_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)
    

def robot_root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.body_ang_vel_w[:,0].reshape(env.num_envs, -1)



def robot_body_ori_w_quat(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.body_quat_w.reshape(env.num_envs, -1)


def robot_body_pos_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # import ipdb; ipdb.set_trace() 

    return asset.data.body_pos_w.reshape(env.num_envs, -1)


def robot_body_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w.reshape(env.num_envs, -1)


def robot_body_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    return asset.data.body_ang_vel_w.reshape(env.num_envs, -1)

# for true state
def ref_body_quat(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # import ipdb; ipdb.set_trace() 
    
    return command.ref_quat_w.reshape(env.num_envs, -1)

def ref_body_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_pos_w.reshape(env.num_envs, -1)

def ref_joint_pos(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_pos.reshape(env.num_envs, -1)

def ref_joint_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.joint_vel.reshape(env.num_envs, -1)

def ref_lin_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_lin_vel_w.reshape(env.num_envs, -1)

def ref_ang_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.ref_ang_vel_w.reshape(env.num_envs, -1)


def default_joint_pos(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    # import ipdb; ipdb.set_trace() 

    return asset.data.default_joint_pos


# ── Clip phase observation ────────────────────────────────────────────────────

def clip_phase(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Normalized progress through current clip: 0.0 = start, 1.0 = end. Shape (num_envs, 1)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.clip_phase


def time_to_live(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Seconds remaining in current motion clip. Shape (num_envs, 1)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_to_live


# ── Future reference motion observations ──────────────────────────────────────

def future_ref_joint_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference joint positions as deltas from current ref joint_pos.

    Returns (num_envs, num_future_steps * num_joints) tensor.
    Each future frame's joint_pos is expressed as (future - current) so the
    policy sees the *change* in joint targets, which is more informative
    than absolute values.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = command.get_future_ref_frames()
    current_joint_pos = command.joint_pos  # (num_envs, num_joints)
    parts = []
    for joint_pos_future, _, _ in future_frames:
        parts.append(joint_pos_future - current_joint_pos)
    return torch.cat(parts, dim=-1)


def future_ref_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Future reference body positions in robot's local frame.

    Returns (num_envs, num_future_steps * num_bodies * 3) tensor.
    Each future frame's body positions are transformed into the current
    robot anchor's local frame (position + orientation).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_frames = command.get_future_ref_frames()
    anchor_pos = command.robot_anchor_pos_w  # (num_envs, 3)
    anchor_quat = command.robot_anchor_quat_w  # (num_envs, 4)
    anchor_quat_inv = quat_inv(anchor_quat)

    parts = []
    for _, body_pos_w_future, _ in future_frames:
        # Transform to local frame: R^-1 * (p_future - p_anchor)
        diff = body_pos_w_future - anchor_pos[:, None, :]
        local_pos = quat_apply(
            anchor_quat_inv[:, None, :].expand_as(diff),
            diff,
        )
        parts.append(local_pos.reshape(env.num_envs, -1))
    return torch.cat(parts, dim=-1)


# ── Privileged "expert" observations ─────────────────────────────────────────
# These obs are added to the `expert` obs group (Phase 2). They expose
# information the deployable policy doesn't get to see (contact forces,
# explicit tracking-error vectors) so the expert can converge faster on the
# privileged-PPO training stage and then DAgger-distill into the student.

def contact_force_mag(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Per-body (e.g. per-foot) contact-force magnitude.

    Shape (num_envs, num_bodies_in_sensor_cfg). Built from the contact sensor's
    `net_forces_w`. Body order follows `sensor_cfg.body_ids`. For the standard
    feet pair, return shape is (num_envs, 2): [left_foot, right_foot].
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]  # (N, K, 3)
    return forces.norm(dim=-1)


def contact_air_time(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Per-body current air time (seconds in the air since last contact).

    Same shape convention as `contact_force_mag`. Useful as a phase signal for
    foot-contact-aware experts.
    """
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.current_air_time[:, sensor_cfg.body_ids]


def body_pos_err_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-body position error (ref − robot) in the pelvis frame.

    Returns (num_envs, num_tracked_bodies * 3). The diff is computed in the
    body-frame `body_pos_relative_w` representation built by the motion
    command (already in pelvis frame post-`_update_command`).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    err = command.body_pos_relative_w - command.robot_body_pos_w  # (N, B, 3)
    return err.reshape(env.num_envs, -1)


def body_ori_err_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-body orientation error as the first 2 cols of (ref_quat * robot_quat^-1)
    rotation matrix in pelvis frame. Returns (num_envs, num_tracked_bodies * 6).

    Using rot6d (first 2 cols of rot matrix) is the same convention as
    `motion_anchor_ori_b` / `robot_body_ori_b`; gives a continuous,
    differentiable error signal."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Error rotation: R_err = R_ref @ R_robot^T  ↔  q_err = q_ref * q_robot^-1
    q_err = quat_mul(command.body_quat_relative_w, quat_inv(command.robot_body_quat_w))
    mat = matrix_from_quat(q_err.reshape(-1, 4))
    return mat[..., :2].reshape(env.num_envs, -1)


def body_lin_vel_err(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-body linear-velocity error (ref - robot) in world frame.
    Returns (num_envs, num_tracked_bodies * 3)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    err = command.body_lin_vel_w - command.robot_body_lin_vel_w
    return err.reshape(env.num_envs, -1)


def body_ang_vel_err(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Per-body angular-velocity error (ref - robot) in world frame.
    Returns (num_envs, num_tracked_bodies * 3)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    err = command.body_ang_vel_w - command.robot_body_ang_vel_w
    return err.reshape(env.num_envs, -1)

# --- Obs-augmentation experiment terms (tracked bodies, env-origin-relative) ---
def robot_body_pos_env(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked-body positions relative to this env's origin (translation-invariant across envs).

    Uses the command's tracked bodies (cfg.body_names) and subtracts the per-env origin, so a
    robot at its env origin reads the same coordinates regardless of which env it is in.
    Shape (num_envs, num_tracked_bodies*3).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos_env = command.robot_body_pos_w - env.scene.env_origins[:, None, :]
    return pos_env.reshape(env.num_envs, -1)


def robot_body_lin_vel_tracked(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Tracked-body world-frame linear velocities. Shape (num_envs, num_tracked_bodies*3)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_body_lin_vel_w.reshape(env.num_envs, -1)


def time_left(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Normalized fraction of the CURRENT clip remaining: 1.0 at clip start, 0.0 at clip end.

    For the multi-clip command, `time_steps` is a GLOBAL index into the concatenation
    of all clips and `motion.time_step_total` is the total length of that concatenation,
    so neither can yield a per-clip fraction on its own. `clip_phase` already computes
    per-clip progress from the per-env `clip_start`/`clip_end` buffers; time-left is its
    complement. Shape (num_envs, 1)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return (1.0 - command.clip_phase).clamp(0.0, 1.0)