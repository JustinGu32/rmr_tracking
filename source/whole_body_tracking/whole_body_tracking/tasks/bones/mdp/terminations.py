from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.bones.mdp.commands import MotionCommand
from whole_body_tracking.tasks.bones.mdp.rewards import _get_body_indexes


def motion_ended(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate when the current motion has ended (reached the last frame).

    Works with MotionCommand (single motion) and MultiMotionCommand/BindedMultiMotionCommand
    (multi-motion with per-env motion_ids). Uses time_steps and motion bounds from the command.

    Note: Terminations are checked before command_manager.compute(), so we check for
    time_steps >= motion_end - 1 (on last frame) since the increment happens in compute().
    """
    command = env.command_manager.get_term(command_name)
    time_steps = command.time_steps

    if hasattr(command, "motion_ids") and hasattr(command.motion, "motion_end_idx"):
        # MultiMotionCommand or BindedMultiMotionCommand: per-env motion end
        motion_end = command.motion.motion_end_idx[command.motion_ids]
        return time_steps >= motion_end - 1
    else:
        # MotionCommand: single motion
        return time_steps >= command.motion.time_step_total - 1


def my_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Terminate on episode timeout OR when the motion clip has ended.

    Combines both conditions under a single time_out=True DoneTerm so that
    the RL algorithm bootstraps value (rather than treating motion completion
    as a hard failure).

    Works with MotionCommand (single motion) and MultiMotionCommand/BindedMultiMotionCommand
    (multi-motion with per-env motion_ids).
    """
    command = env.command_manager.get_term(command_name)
    time_steps = command.time_steps

    episode_length_term = env.episode_length_buf >= env.max_episode_length

    if hasattr(command, "motion_ids") and hasattr(command.motion, "motion_end_idx"):
        motion_end = command.motion.motion_end_idx[command.motion_ids]
        max_step_length_term = time_steps >= motion_end - 1
    else:
        max_step_length_term = time_steps >= command.motion.time_step_total - 1

    return episode_length_term | max_step_length_term


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_pos_x_y_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    err = torch.norm(command.anchor_pos_w[:, 0:2] - command.robot_anchor_pos_w[:, 0:2], dim=1)
    return err > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
