from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude, quat_inv, quat_apply, quat_mul

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

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
    robot_anchor_pos = command.robot_anchor_pos_w
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_pos_rel = command.robot_vr_3point_pos_w - robot_anchor_pos[:, None, :]
    vr_pos_rel = quat_apply(robot_anchor_quat_inv, vr_pos_rel)
    ref_anchor_pos = command.anchor_pos_w
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
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
    robot_anchor_quat_inv = quat_inv(command.robot_anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
    vr_quat_rel = quat_mul(robot_anchor_quat_inv, command.vr_3point_body_quat_w)
    ref_anchor_quat_inv = quat_inv(command.anchor_quat_w).view(env.num_envs, 1, 4).repeat(1, 3, 1)
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

def motion_joint_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.mean(torch.square(command.joint_pos - command.robot_joint_pos), dim=-1)
    return torch.exp(-error.mean(-1) / std**2)

def double_step_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    moving_threshold: float,
    stance_vel_threshold: float = 0.3,
    body_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize when the robot moves a foot that the reference motion has planted.

    A double step is when reference foot speed < stance_vel_threshold (foot should be planted)
    but robot foot speed > moving_threshold (robot lifts it anyway).
    Returns -1 per offending foot, averaged across feet.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    # (num_envs, num_feet)
    ref_foot_speed = torch.norm(command.body_lin_vel_w[:, body_indexes], dim=-1)
    robot_foot_speed = torch.norm(command.robot_body_lin_vel_w[:, body_indexes], dim=-1)
    ref_in_stance = ref_foot_speed < stance_vel_threshold
    robot_moving = robot_foot_speed > moving_threshold
    return -(ref_in_stance & robot_moving).float().mean(dim=-1)


def _log_to_extras(env, key: str, value: float) -> None:
    """Write a scalar into env.extras['log'] so the runner logs it to wandb each step."""
    env.extras.setdefault("log", {})[key] = value


def _ref_stance_mask(
    command: MotionCommand,
    body_indexes: list[int],
    stance_vel_threshold: float,
) -> torch.Tensor:
    """Return (num_envs, num_feet) bool: reference foot is planted."""
    ref_foot_speed = torch.norm(command.body_lin_vel_w[:, body_indexes], dim=-1)
    return ref_foot_speed < stance_vel_threshold


def _robot_contact_mask(
    contact_sensor,
    foot_body_ids,
    contact_force_threshold: float,
) -> torch.Tensor:
    """Return (num_envs, num_feet) bool: robot foot has ground contact."""
    # net_forces_w_history: (num_envs, history, num_bodies, 3)
    forces = contact_sensor.data.net_forces_w_history[:, :, foot_body_ids, :]
    force_mag = torch.norm(forces, dim=-1)               # (num_envs, history, num_feet)
    return force_mag.max(dim=1).values > contact_force_threshold  # (num_envs, num_feet)


def stance_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    body_names: list[str] | None = None,
    stance_vel_threshold: float = 0.25,
    contact_force_threshold: float = 10.0,
) -> torch.Tensor:
    """Penalize when the reference foot is planted but the robot foot has no contact.

    Directly captures "robot lifted a foot that should be on the ground",
    which is more reliable than the velocity-only check in double_step_penalty.
    Returns -1 per offending foot, summed across feet.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_indexes = _get_body_indexes(command, body_names)

    ref_stance = _ref_stance_mask(command, body_indexes, stance_vel_threshold)
    robot_contact = _robot_contact_mask(contact_sensor, sensor_cfg.body_ids, contact_force_threshold)
    missing_contact = ref_stance & ~robot_contact

    _log_to_extras(env, "stance/ref_stance_frac", float(ref_stance.float().mean().item()))
    _log_to_extras(env, "stance/contact_penalty_active_frac", float(missing_contact.float().mean().item()))
    return -missing_contact.float().sum(dim=-1)


def stance_slide_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    body_names: list[str] | None = None,
    stance_vel_threshold: float = 0.25,
    contact_force_threshold: float = 10.0,
    slide_vel_threshold: float = 0.15,
) -> torch.Tensor:
    """Penalize when the robot foot is contacting but sliding during reference stance.

    Catches the case where the foot is on the ground but still moving laterally —
    a common precursor to or symptom of a double step.
    Returns -1 per offending foot, summed across feet.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_indexes = _get_body_indexes(command, body_names)

    ref_stance = _ref_stance_mask(command, body_indexes, stance_vel_threshold)
    robot_contact = _robot_contact_mask(contact_sensor, sensor_cfg.body_ids, contact_force_threshold)

    # XY speed only — vertical settling motion during contact is fine
    robot_foot_speed_xy = torch.norm(command.robot_body_lin_vel_w[:, body_indexes, :2], dim=-1)
    sliding = ref_stance & robot_contact & (robot_foot_speed_xy > slide_vel_threshold)

    _log_to_extras(env, "stance/slide_penalty_active_frac", float(sliding.float().mean().item()))
    return -sliding.float().sum(dim=-1)


def stance_drift_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    body_names: list[str] | None = None,
    stance_vel_threshold: float = 0.25,
    contact_force_threshold: float = 10.0,
    drift_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize when a stance foot drifts from where it first made contact.

    On each reference stance entry, the current robot foot XY position is saved.
    While reference stance continues, any XY drift beyond drift_threshold is penalized.
    This directly attacks double-stepping: even without a full foot lift, repositioning
    the planted foot is caught and penalized.

    Persistent per-env buffers are stored on env:
        env._stance_prev_ref_stance  (num_envs, num_feet)  bool
        env._stance_saved_pos        (num_envs, num_feet, 3) float
    Both are reset for any env whose episode_length_buf <= 1.

    Returns -1 per offending foot, summed across feet.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    body_indexes = _get_body_indexes(command, body_names)
    num_feet = len(body_indexes)

    ref_stance = _ref_stance_mask(command, body_indexes, stance_vel_threshold)
    robot_contact = _robot_contact_mask(contact_sensor, sensor_cfg.body_ids, contact_force_threshold)
    robot_foot_pos = command.robot_body_pos_w[:, body_indexes, :3]  # (N, num_feet, 3)

    # ── lazy-init persistent buffers ──────────────────────────────────────
    if not hasattr(env, "_stance_prev_ref_stance"):
        env._stance_prev_ref_stance = torch.zeros(
            env.num_envs, num_feet, dtype=torch.bool, device=env.device
        )
        env._stance_saved_pos = robot_foot_pos.detach().clone()

    # ── reset buffers for envs that just started a new episode ────────────
    just_reset = (env.episode_length_buf <= 1)  # (num_envs,)
    if just_reset.any():
        env._stance_prev_ref_stance[just_reset] = False
        env._stance_saved_pos[just_reset] = robot_foot_pos[just_reset].detach()

    # ── save foot position at the moment each foot enters stance ──────────
    stance_start = ref_stance & ~env._stance_prev_ref_stance  # (N, num_feet)
    if stance_start.any():
        env._stance_saved_pos[stance_start] = robot_foot_pos[stance_start].detach()

    # ── penalize XY drift from saved contact point ────────────────────────
    drift_xy = torch.norm(
        robot_foot_pos[:, :, :2] - env._stance_saved_pos[:, :, :2], dim=-1
    )  # (N, num_feet)

    bad_drift = ref_stance & robot_contact & (drift_xy > drift_threshold)

    _log_to_extras(env, "stance/drift_penalty_active_frac", float(bad_drift.float().mean().item()))
    _log_to_extras(env, "stance/mean_drift_xy_m", float(drift_xy[ref_stance].mean().item() if ref_stance.any() else 0.0))

    # ── advance buffers ───────────────────────────────────────────────────
    env._stance_prev_ref_stance.copy_(ref_stance)

    return -bad_drift.float().sum(dim=-1)
