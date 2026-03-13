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
# Interpolation helper (mirrors dexsuite's initial_final_interpolate_fn)
# ---------------------------------------------------------------------------

def initial_final_interpolate_fn(env: ManagerBasedRLEnv, env_id, data, initial_value, final_value, difficulty_term_str):
    """Interpolate between initial_value and final_value based on difficulty_frac.

    Supports arbitrarily nested lists/tuples. Scalars are interpolated at the leaves.
    """
    difficulty_term: AssistiveForceScheduler = getattr(env.curriculum_manager.cfg, difficulty_term_str).func
    frac = difficulty_term.difficulty_frac
    if frac < 0.1:
        return mdp.modify_env_param.NO_CHANGE

    initial_value_tensor = torch.tensor(initial_value, device=env.device)
    final_value_tensor = torch.tensor(final_value, device=env.device)

    return _recurse(initial_value_tensor.tolist(), final_value_tensor.tolist(), data, frac)


def _recurse(iv_elem, fv_elem, data_elem, frac):
    if isinstance(data_elem, Sequence) and not isinstance(data_elem, (str, bytes)):
        return type(data_elem)(_recurse(iv_e, fv_e, d_e, frac) for iv_e, fv_e, d_e in zip(iv_elem, fv_elem, data_elem))
    new_val = frac * (fv_elem - iv_elem) + iv_elem
    if isinstance(data_elem, int):
        return int(new_val)
    return float(new_val)


# ---------------------------------------------------------------------------
# Assistive-force curriculum interpolation (modify_fn for modify_term_cfg)
# ---------------------------------------------------------------------------

# def assistive_force_interpolate_fn(
#     env: ManagerBasedRLEnv,
#     env_ids,
#     data,
#     difficulty_term_str,
#     cutoff_steps: int = 240000,
# ):
#    """Interpolate curriculum_factor from 1.0 → 0.0 based on ADR difficulty,
#    with a hard cutoff after ``cutoff_steps`` physics steps.

#    Option A (active): ADR-based scheduling with hard cutoff.
#      - Before cutoff: curriculum_factor = 1 - difficulty_frac
#      - After cutoff:  curriculum_factor = 0  (forces fully off)

#    Args:
#        cutoff_steps: Physics step count after which forces are hard-zeroed.
#                      Default 240000 = 10k iters × 24 steps_per_env.
#    """
    # Hard cutoff: force fully off after cutoff_steps
#    if env.common_step_counter >= cutoff_steps:
#        return 0.0

    # Before cutoff: ADR-based scheduling
#    difficulty_term: AssistiveForceScheduler = getattr(
#        env.curriculum_manager.cfg, difficulty_term_str
#    ).func
#    frac = difficulty_term.difficulty_frac

#    new_factor = 1.0 - frac  # full assist at difficulty 0, no assist at max
#    return float(new_factor)


# --- Option B (uncomment to use instead of Option A above) ---
# Gradual ramp-to-zero safety net: if ADR hasn't fully removed forces
# by ramp_start_steps, linearly ramp whatever remains to 0 by cutoff_steps.
# This avoids any cliff even if ADR is slow to reach max difficulty.
#
def assistive_force_interpolate_fn(
    env: ManagerBasedRLEnv,
    env_ids,
    data,
    difficulty_term_str,
    cutoff_steps: int = 240000,
    ramp_start_steps: int = 168000,  # start safety ramp at ~7k iters (70% of cutoff)
):
    """ADR-based scheduling with gradual ramp-to-zero safety net.

    - Before ramp_start: curriculum_factor = 1 - difficulty_frac (pure ADR)
    - ramp_start → cutoff: linearly blend ADR factor toward 0
    - After cutoff: curriculum_factor = 0  (forces fully off)
    Args:
        cutoff_steps: Physics step count after which forces are hard-zeroed.
        ramp_start_steps: Step at which the safety ramp begins blending toward 0.
    """
    if env.common_step_counter >= cutoff_steps:
        return 0.0

    difficulty_term: AssistiveForceScheduler = getattr(
        env.curriculum_manager.cfg, difficulty_term_str
    ).func
    frac = difficulty_term.difficulty_frac
    adr_factor = 1.0 - frac

    if env.common_step_counter >= ramp_start_steps:
        # Safety ramp: linearly blend adr_factor → 0 over remaining window
        ramp_progress = (env.common_step_counter - ramp_start_steps) / (cutoff_steps - ramp_start_steps)
        adr_factor = adr_factor * (1.0 - ramp_progress)

    return float(adr_factor)


# ---------------------------------------------------------------------------
# Adaptive difficulty scheduler (mirrors dexsuite's DifficultyScheduler)
# ---------------------------------------------------------------------------

class AssistiveForceScheduler(ManagerTermBase):
    """Adaptive scheduler that reduces assistance as motion tracking improves.

    Tracks per-environment difficulty levels. When anchor position tracking error
    falls below ``pos_tol``, difficulty increases (less assistance). Otherwise it
    decreases (more assistance). The normalised mean difficulty is exposed as
    ``difficulty_frac`` for use in curriculum interpolation terms.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        init_difficulty = self.cfg.params.get("init_difficulty", 0)
        self.current_adr_difficulties = torch.ones(env.num_envs, device=env.device) * init_difficulty
        self.difficulty_frac = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        command_name: str = "motion",
        pos_tol: float = 0.15,
        init_difficulty: int = 0,
        min_difficulty: int = 0,
        max_difficulty: int = 10,
    ):
        command = env.command_manager.get_term(command_name)
        pos_err = command.metrics["error_anchor_pos"][env_ids]

        move_up = pos_err < pos_tol  # tracking well → increase difficulty (reduce assist)
        self.current_adr_difficulties[env_ids] = torch.where(
            move_up,
            self.current_adr_difficulties[env_ids] + 1,
            self.current_adr_difficulties[env_ids] - 1,
        ).clamp(min=min_difficulty, max=max_difficulty)

        self.difficulty_frac = torch.mean(self.current_adr_difficulties) / max(max_difficulty, 1)
        return self.difficulty_frac


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

