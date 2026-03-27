from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_inv, quat_apply, quat_mul

from whole_body_tracking.tasks.bones.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def vr_position_relative_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_points = len(command.cfg.vr_3point_body)
    robot_anchor_pos = command.robot_anchor_pos_w
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, n_points, 1)
    vr_pos_rel = command.robot_vr_3point_pos_w - robot_anchor_pos[:, None, :]
    vr_pos_rel = quat_apply(robot_anchor_quat_inv, vr_pos_rel)
    ref_anchor_pos = command.anchor_pos_w
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, n_points, 1)
    vr_pos_rel_ref = command.vr_3point_body_pos_w - ref_anchor_pos[:, None, :]
    vr_pos_rel_ref = quat_apply(ref_anchor_quat_inv, vr_pos_rel_ref)

    error = torch.sum(
        torch.square(vr_pos_rel_ref - vr_pos_rel), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)

def vr_orientation_relative_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    n_points = len(command.cfg.vr_3point_body)
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, n_points, 1)
    vr_quat_rel = quat_mul(robot_anchor_quat_inv, command.robot_vr_3point_quat_w)
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, n_points, 1)
    vr_quat_rel_ref = quat_mul(ref_anchor_quat_inv, command.vr_3point_body_quat_w)

    error = (
        quat_error_magnitude(vr_quat_rel_ref, vr_quat_rel)
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)

def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def foot_contact_state_penalty(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, height_threshold: float
) -> torch.Tensor:
    """Penalize foot being on the ground when reference motion says it should be in the air.

    Triggers when reference ankle z > height_threshold (foot should be lifted)
    AND the robot's foot is in contact with the ground (from contact sensor).
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Reference ankle world z-heights
    ankle_indexes = _get_body_indexes(command, ["left_ankle_roll_link", "right_ankle_roll_link"])
    ref_ankle_z = command.body_pos_relative_w[:, ankle_indexes, 2]  # (N, 2)

    # Foot should be in the air if reference ankle z is above ground + threshold
    should_be_in_air = ref_ankle_z > height_threshold  # (N, 2)

    # Check actual contact from sensor
    net_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]  # (N, 2, 3)
    is_in_contact = torch.norm(net_forces, dim=-1) > 1.0  # (N, 2)

    # Penalize: foot on ground when it should be in air
    violation = (should_be_in_air & is_in_contact).float()
    return -violation.sum(dim=-1)


def grasp_contact_reward(env: ManagerBasedRLEnv, command_name: str, distance_threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    env_should_grasp = command.wrist_grasp_label
    
    # virtual desired contact locations
    wrist_pos_ref = command.vr_3point_body_pos_w[:,:2]
    root_pos_ref = command.anchor_pos_w
    root_orn_ref = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    robot_root_pos = command.robot_anchor_pos_w
    robot_root_orn = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    wrist_pos_ref_l = wrist_pos_ref - root_pos_ref[:,None,:]
    wrist_pos_ref_l = quat_apply(quat_inv(root_orn_ref), wrist_pos_ref_l)
    wrist_pos_ref_wr = quat_apply(robot_root_orn, wrist_pos_ref_l) + robot_root_pos[:,None,:]

    robot_wrist_pos = command.robot_vr_3point_pos_w[:,:2]
    o2r = robot_wrist_pos - wrist_pos_ref_wr
    dist2o = torch.norm(o2r, dim=-1)
    left_correct_contact_mask = (dist2o[:,0] < distance_threshold) * env_should_grasp
    right_correct_contact_mask = (dist2o[:,1] < distance_threshold) * env_should_grasp
    reward = left_correct_contact_mask.float() + right_correct_contact_mask.float()
    return reward