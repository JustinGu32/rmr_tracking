from __future__ import annotations

import torch
from isaaclab.utils.math import quat_mul, quat_inv, axis_angle_from_quat
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




# ---------------------------------------------------------------------------
# Spring force toward reference motion (replaces simple Z-upward force)
# ---------------------------------------------------------------------------

def apply_spring_force(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_name: str = "robot",
    stiffness: float = 500.0,
    ang_stiffness: float = 50.0,
    damping: float = 20.0,
    axis_weights: tuple[float, float, float] = (0.3, 0.3, 5.0),
    gravity_comp: float = 0.0,
    curriculum_factor: float = 1.0,
    env_ids: torch.Tensor | None = None,
):
    """Apply a PD spring force pulling the robot's anchor body toward the reference motion.

    The force is computed as:
        F_spring = axis_weights * stiffness * (ref_pos - cur_pos)
                 + damping * (ref_vel - cur_vel)
                 + ang_stiffness * axis_angle_error(ref_quat, cur_quat)
        F_total  = F_spring * curriculum_factor

    Called every step from StaircaseEnv.step() before physics stepping.

    Args:
        env: The environment.
        command_name: Name of the motion command term.
        asset_name: Name of the robot asset in the scene.
        stiffness: Spring stiffness (Kp).
        ang_stiffness: Angular spring stiffness (Kp_ang).
        damping: Velocity damping (Kd).
        gravity_comp: Fraction of gravity to compensate (0.0–1.0).
        axis_weights: Per-axis multiplier on spring stiffness [x, y, z].
        curriculum_factor: Scaling factor from curriculum (1.0 = full assist, 0.0 = none).
        env_ids: Optional tensor of env indices to apply force to. If None, applies to all.
    """
    asset: Articulation = env.scene[asset_name]
    command = env.command_manager.get_term(command_name)
    anchor_idx = command.robot_anchor_body_index

    # Reference anchor state (world frame, includes env origins)
    ref_pos_w = command.anchor_pos_w          # (num_envs, 3)
    ref_vel_w = command.anchor_lin_vel_w      # (num_envs, 3)
    ref_quat_w = command.anchor_quat_w        # (num_envs, 4)

    # Current robot anchor state (world frame)
    cur_pos_w = asset.data.body_pos_w[:, anchor_idx, :]      # (num_envs, 3)
    cur_vel_w = asset.data.body_lin_vel_w[:, anchor_idx, :]   # (num_envs, 3)
    cur_quat_w = asset.data.body_quat_w[:, anchor_idx, :]    # (num_envs, 4)

    # Position and velocity errors
    pos_error = ref_pos_w - cur_pos_w   # (num_envs, 3)
    vel_error = ref_vel_w - cur_vel_w   # (num_envs, 3)

    # Per-axis weighted spring force
    weights = torch.tensor(axis_weights, device=env.device)  # (3,)
    spring_force = weights * stiffness * pos_error + damping * vel_error  # (num_envs, 3)

    # Unilateral Z-axis: only push up, never pull down
    spring_force[:, 2] = torch.clamp(spring_force[:, 2], min=0.0)

    # Angular spring torque: Kp_ang * axis_angle_error(ref_quat, cur_quat)
    quat_error = quat_mul(ref_quat_w, quat_inv(cur_quat_w))  # (num_envs, 4)
    ang_error = axis_angle_from_quat(quat_error)              # (num_envs, 3)
    spring_torque = weights * ang_stiffness * ang_error        # (num_envs, 3)

    # Gravity compensation (opt-in, default off for training)
    grav_force = torch.zeros_like(spring_force)
    if gravity_comp > 0.0:
        total_mass = asset.root_physx_view.get_masses().sum(dim=1)  # (num_envs,)
        grav_force[:, 2] = total_mass * 9.81 * gravity_comp

    # Total force & torque, scaled by curriculum
    total_force = (spring_force + grav_force) * curriculum_factor
    total_torque = spring_torque * curriculum_factor

    if env_ids is not None:
        # Apply only to specified envs
        forces = total_force[env_ids].unsqueeze(1)   # (len(env_ids), 1, 3)
        torques = total_torque[env_ids].unsqueeze(1)  # (len(env_ids), 1, 3)
        asset.set_external_force_and_torque(
            forces, torques, body_ids=[anchor_idx], env_ids=env_ids, is_global=True
        )
    else:
        # Apply to all envs
        forces = total_force.unsqueeze(1)   # (num_envs, 1, 3)
        torques = total_torque.unsqueeze(1)  # (num_envs, 1, 3)
        asset.set_external_force_and_torque(
            forces, torques, body_ids=[anchor_idx], is_global=True
        )

    return total_force

