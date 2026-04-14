from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv, quat_mul

from whole_body_tracking.tasks.bones.mdp.commands import MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def command_lookahead(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command = MultiMotionCommand = env.command_manager.get_term(command_name)
    return command.command_lookahead

def command_lower_body(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_lower_body

def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)

def projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    grav_dir = quat_apply(quat_inv(command.robot_anchor_quat_w), command.down_dir)
    return grav_dir.view(env.num_envs, -1)

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

    # Clamp feet compliant targets above ground
    if command.vr_3point_ground_clamp_indices:
        ground_z = env.scene.env_origins[:, 2] + 0.005
        for idx in command.vr_3point_ground_clamp_indices:
            compliant_z_w = command.vr_3point_body_pos_w[:, idx, 2] - ext_force_disp_w[:, idx, 2]
            violation = (ground_z - compliant_z_w).clamp(min=0.0)
            correction_w = torch.zeros(env.num_envs, 3, device=command.device)
            correction_w[:, 2] = violation
            correction_l = quat_apply(quat_inv(ref_root_quat[:, idx]), correction_w)
            ref_3point_root[:, idx] += correction_l

    return ref_3point_root.view(env.num_envs, -1)

def vr_3point_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_quat = command.vr_3point_body_quat_w
    ref_3point_root = quat_mul(quat_inv(ref_root_quat),ref_3point_quat)
    return ref_3point_root.view(env.num_envs, -1)

def wrist_grasp_label(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.wrist_grasp_label.view(env.num_envs, -1)

def last_action_pseudotarget(env: ManagerBasedEnv) -> torch.Tensor:
    """Last action in pseudo-target units: (processed_actions - default_pos) / scale.

    For target mode this equals the raw PPO output (backward compatible).
    For delta-pseudotarget mode this equals (x_ref + delta - default_pos) / scale, matching the ONNX output format.
    """
    action_term = env.action_manager.get_term("joint_pos")
    default_pos = env.scene["robot"].data.default_joint_pos
    return (action_term.processed_actions - default_pos) / action_term._scale


def compliance(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.eef_stiffness_buf * 10.0