from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def apply_assistive_forces(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stiffness: float,
    damping: float,
    gravity_comp: float,
    curriculum_factor: float = 1.0,
):
    """
    Apply assistive forces (spring + gravity compensation) to the robot's anchor body.
    
    Args:
        env: The environment.
        command_name: The name of the command term providing the reference motion.
        asset_cfg: The configuration for the asset (robot) to apply forces to.
        stiffness: Spring stiffness (Kp) for position error.
        damping: Damping (Kd) for velocity error.
        gravity_comp: Gravity compensation factor (0.0 to 1.0).
        curriculum_factor: A factor (0.0 to 1.0) scaling the forces. 1.0 = full assist, 0.0 = no assist.
    """
    
    # 0. Get the robot asset
    asset = env.scene[asset_cfg.name]
    
    # 1. Get reference motion
    command = env.command_manager.terms[command_name]
    
    # Reference root state (world frame)
    ref_pos_w = command.anchor_pos_w
    ref_lin_vel_w = command.anchor_lin_vel_w
    
    # 2. Get current robot state (world frame)
    body_idx = command.robot_anchor_body_index
    
    # Check shape compatibility
    # asset.data.body_pos_w is (num_envs, num_bodies, 3)
    current_pos_w = asset.data.body_pos_w[:, body_idx, :]
    current_lin_vel_w = asset.data.body_lin_vel_w[:, body_idx, :]
    
    # 3. Calculate errors
    pos_error = ref_pos_w - current_pos_w
    vel_error = ref_lin_vel_w - current_lin_vel_w
    
    # 4. Calculate Spring Force (PD control)
    # F = (Kp * pos_err + Kd * vel_err) * curriculum_factor
    spring_force = (stiffness * pos_error + damping * vel_error) * curriculum_factor.unsqueeze(-1)
    
    # 5. Calculate Gravity Compensation
    # F_g = mass * g * vector_up * scale * curriculum_factor
    # Sum masses of all links to estimate total mass to lift
    total_mass = asset.root_physx_view.get_masses().sum(dim=1)
    gravity_vec = torch.tensor([0.0, 0.0, 9.81], device=env.device).repeat(env.num_envs, 1)
    
    grav_force = total_mass[:, None] * gravity_vec * gravity_comp * curriculum_factor.unsqueeze(-1)
    
    # 6. Apply Total Force
    total_force = spring_force + grav_force
    
    # Create force tensor for all bodies (num_envs, num_bodies, 3)
    forces = torch.zeros(env.num_envs, asset.num_bodies, 3, device=env.device)
    forces[:, body_idx, :] = total_force
    
    # Apply to simulation (set buffer for next step)
    # Using write_data_to_sim pattern or direct physx call
    # Note: external forces are usually cleared every step by the physics engine unless persistent, 
    # but Isaac Lab's `set_external_force_and_torque` sets the buffer that is applied.
    asset.set_external_force_and_torque(forces=forces)
