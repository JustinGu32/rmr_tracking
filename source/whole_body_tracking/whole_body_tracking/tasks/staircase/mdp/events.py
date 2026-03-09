from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_rigid_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    com_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg,
):
    """Randomize the center of mass (CoM) of rigid bodies by adding a random value sampled from the given ranges.

    .. note::
        This function uses CPU tensors to assign the CoM. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # sample random CoM values
    range_list = [com_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    ranges = torch.tensor(range_list, device="cpu")
    rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 3), device="cpu").unsqueeze(1)

    # get the current com of the bodies (num_assets, num_bodies)
    coms = asset.root_physx_view.get_coms().clone()

    # Randomize the com in range
    coms[:, body_ids, :3] += rand_samples

    # Set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


# ---------------------------------------------------------------------------
# Assistive spring force event (called every step via EventTerm interval)
# ---------------------------------------------------------------------------

def apply_assistive_spring_force(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "motion",
    asset_name: str = "robot",
    stiffness: float = 250.0,
    ang_stiffness: float = 10.0,
    damping: float = 15.0,
    axis_weights: tuple[float, float, float] = (0.5, 0.5, 2.0),
    curriculum_factor: float = 1.0,
):
    """Apply PD spring force toward the reference motion, scaled by curriculum_factor.

    This is an event function wrapper around ``apply_spring_force`` in
    ``curriculum.py``.  The ``curriculum_factor`` parameter starts at 1.0
    (full assistance) and is reduced toward 0.0 by the ADR curriculum via
    ``modify_term_cfg`` at each episode reset.

    Args:
        env: The environment.
        env_ids: Environment indices (unused — force applied to all envs).
        command_name: Name of the motion command term.
        asset_name: Name of the robot asset in the scene.
        stiffness: Spring stiffness (Kp).
        ang_stiffness: Angular spring stiffness (Kp_ang).
        damping: Velocity damping (Kd).
        axis_weights: Per-axis multiplier on spring stiffness [x, y, z].
        curriculum_factor: Scaling factor from curriculum (1.0 = full assist, 0.0 = none).
    """
    from .curriculum import apply_spring_force

    # Store curriculum_factor and last force on env for logging in _reset_idx
    env._spring_force_curriculum_factor = curriculum_factor

    if curriculum_factor <= 0.0:
        # Force fully ramped down — zero out any persistent force buffer
        if getattr(env, "_spring_force_active", True):
            env._spring_force_active = False
            env._last_spring_force = None
            robot = env.scene[asset_name]
            anchor_idx = env.command_manager.get_term(command_name).robot_anchor_body_index
            zero_force = torch.zeros(env.num_envs, 1, 3, device=env.device)
            robot.set_external_force_and_torque(
                zero_force, zero_force, body_ids=[anchor_idx], is_global=True
            )
        return

    env._spring_force_active = True
    env._last_spring_force = apply_spring_force(
        env=env,
        command_name=command_name,
        asset_name=asset_name,
        stiffness=stiffness,
        ang_stiffness=ang_stiffness,
        damping=damping,
        axis_weights=axis_weights,
        curriculum_factor=curriculum_factor,
    )
