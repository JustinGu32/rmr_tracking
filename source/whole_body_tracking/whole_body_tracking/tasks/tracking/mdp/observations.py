from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv, quat_mul

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

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
