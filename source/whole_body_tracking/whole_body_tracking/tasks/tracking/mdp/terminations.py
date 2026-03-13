from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand
from whole_body_tracking.tasks.tracking.mdp.rewards import _get_body_indexes
# TODO: currently single motion tracking, change to concatenate additional motions
def my_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # import ipdb; ipdb.set_trace() 

    episode_length_term = env.episode_length_buf >= env.max_episode_length
    max_step_length_term = command.time_steps >= command.motion.time_step_total-1
    # print("TIMEOUT: ", (episode_length_term | max_step_length_term))
    return episode_length_term | max_step_length_term


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # print("BAD_ANCHOR_POS: ", (torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold))
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_x_y_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    err = torch.norm(command.anchor_pos_w[:, 0:2] - command.robot_anchor_pos_w[:, 0:2], dim=1)
    if err[0] > threshold:
        print(f"[TERM] anchor_pos_w={command.anchor_pos_w[0].tolist()}, robot={command.robot_anchor_pos_w[0].tolist()}, err={err[0]:.4f}, time_step={command.time_steps[0].item()}")
    # print("BAD ANCHOR POS XY: ", (err > threshold))
    return err > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # print("BAD ANCHOR POS Z", torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    # print("BAD ANCHOR ORI: ", ((motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold))
    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes], dim=-1)
    # print("BAD MOTION BODY POS: ", (torch.any(error > threshold, dim=-1)))
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    # print("BAD MOTION BODY POS Z: ", (torch.any(error > threshold, dim=-1)))
    return torch.any(error > threshold, dim=-1)


def double_step(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    """Terminate when any foot's squared velocity error exceeds a threshold (double-stepping)."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.any(error > threshold, dim=-1)
