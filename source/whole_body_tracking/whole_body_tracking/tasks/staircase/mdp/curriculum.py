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
# Linear decay scheduler for assistive forces
# ---------------------------------------------------------------------------
class LinearForceScheduler(ManagerTermBase):
    """Linearly decays assistance from 1.0 to 0.0 based on environment steps."""
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.difficulty_frac = 0.0  # 0.0 = max assist, 1.0 = min assist (forces off)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "motion",
        start_steps: int = 120000,
        ramp_steps: int = 240000,
    ):
        if env.common_step_counter <= start_steps:
            self.difficulty_frac = 0.0
        else:
            t = (env.common_step_counter - start_steps) / max(ramp_steps - start_steps, 1)
            self.difficulty_frac = min(float(t), 1.0)
            
        return self.difficulty_frac

# ---------------------------------------------------------------------------
# Interpolation helper (mirrors dexsuite's initial_final_interpolate_fn)
# ---------------------------------------------------------------------------

def linear_interpolate_fn(env: ManagerBasedRLEnv, env_ids, data, initial_value, final_value, difficulty_term_str):
    """Interpolate between initial_value and final_value based on difficulty_frac."""
    difficulty_term: LinearForceScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)
    # _recurse allows arbitrarily nested lists/tuples
    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)



def _recurse(iv_elem, fv_elem, data_elem, frac):
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val)
    return float(new_val)





# ---------------------------------------------------------------------------
# Spring force toward reference motion (replaces simple Z-upward force)
# ---------------------------------------------------------------------------

def apply_spring_force(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    asset_name: str = "robot",
    stiffness: float = 500.0,
    ang_stiffness: float = 50.0,
    damping: float = 20.0,
    axis_weights: tuple[float, float, float] = (0.3, 0.3, 5.0),
    gravity_comp: float = 0.0,
    curriculum_factor: float | None = None,
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
        env_ids: Optional tensor of env indices to apply force to. If None, applies to all.
        command_name: Name of the motion command term.
        asset_name: Name of the robot asset in the scene.
        stiffness: Spring stiffness (Kp).
        ang_stiffness: Angular spring stiffness (Kp_ang).
        damping: Velocity damping (Kd).
        gravity_comp: Fraction of gravity to compensate (0.0–1.0).
        axis_weights: Per-axis multiplier on spring stiffness [x, y, z].
        curriculum_factor: Scaling factor. If None, checks env._spring_force_curriculum_factor.
    """
    if curriculum_factor is None:
        curriculum_factor = getattr(env, "_spring_force_curriculum_factor", 1.0)

    # Keep env-side state in sync for logging (read in StaircaseEnv._reset_idx).
    if torch.is_tensor(curriculum_factor):
        curriculum_factor_log = float(curriculum_factor.mean().item())
    else:
        curriculum_factor_log = float(curriculum_factor)
    env._spring_force_curriculum_factor = curriculum_factor_log
    env._spring_force_active = curriculum_factor_log > 0.0
        
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
    # Use max(0, force) to ensure no negative force is applied when descending
    # spring_force = torch.clamp(spring_force, min=0.0)

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
    spring_force_scaled = spring_force * curriculum_factor
    grav_force_scaled = grav_force * curriculum_factor

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

    if env._spring_force_active:
        env._last_total_assist_force = total_force
        env._last_spring_force = spring_force_scaled
        env._last_grav_force = grav_force_scaled
        env._last_spring_torque = total_torque
    else:
        env._last_total_assist_force = None
        env._last_spring_force = None
        env._last_grav_force = None
        env._last_spring_torque = None

    return total_force
    
