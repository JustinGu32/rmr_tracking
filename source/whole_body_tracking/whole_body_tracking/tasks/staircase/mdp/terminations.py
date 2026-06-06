from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

# Isaac Lab v2.1.0 has quat_rotate_inverse; newer versions renamed/added quat_apply_inverse.
# They are mathematically identical (rotate vec by quat^-1).
_quat_apply_inverse = getattr(math_utils, "quat_apply_inverse", None) or math_utils.quat_rotate_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from whole_body_tracking.tasks.staircase.mdp.commands import MotionCommand
from whole_body_tracking.tasks.staircase.mdp.rewards import _get_body_indexes
# TODO: currently single motion tracking, change to concatenate additional motions
def my_time_out(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # import ipdb; ipdb.set_trace() 

    episode_length_term = env.episode_length_buf >= env.max_episode_length
    max_step_length_term = command.time_steps >= command.motion.time_step_total-1

    return episode_length_term | max_step_length_term


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = _quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = _quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

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


def bad_stair_phase(
    env: ManagerBasedRLEnv, command_name: str, grace: int = 5, min_steps: int = 15
) -> torch.Tensor:
    """Terminate when a foot is not on the stair the reference expects it on, for too long.

    Teaches correct stepping: at each frame the reference says which stair each foot should be
    planted on (anywhere on that stair). If a foot stays OFF its expected stair for more than
    ``grace`` consecutive steps, terminate. A ``min_steps`` warmup skips this check at the very
    start of an episode so a slightly-off / perturbed reset doesn't instantly terminate.

    Relies on MotionCommand precomputing the schedule (cfg.stair_bounds set) and maintaining
    ``foot_off_streak`` each step. If the schedule is disabled, never terminates.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    if getattr(command, "stair_expected", None) is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    over_grace = (command.foot_off_streak > grace).any(dim=-1)
    past_warmup = env.episode_length_buf >= min_steps
    return over_grace & past_warmup
