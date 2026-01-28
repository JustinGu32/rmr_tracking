from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

from isaaclab.assets import Articulation, DeformableObject, RigidObject

from isaaclab.utils.math import quat_mul, quat_apply, quat_inv, quat_error_magnitude, yaw_quat, sample_uniform, quat_from_euler_xyz


from whole_body_tracking.tasks.tracking.mdp.commands import MotionCommand

def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]

def apply_random_body_forces(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    body_names: list[str],
    asset_cfg: SceneEntityCfg,
    force_std: tuple[float, float, float] = (5.0, 5.0, 5.0),
    torque_std: tuple[float, float, float] = (1.0, 1.0, 1.0),
):
    """Apply random forces to specific bodies."""
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    body_indexes = _get_body_indexes(command, body_names)
    body_indexes_tensor = torch.tensor(body_indexes, dtype=torch.long, device=env.device)
    
    num_bodies = len(body_indexes)
    

    num_target_envs = len(env_ids)
    num_target_bodies = len(body_indexes)

    # 3. Create the "small" tensors of random values
    # Shape: (num_target_envs, num_target_bodies, 3)
    forces = torch.randn(num_target_envs, num_target_bodies, 3, device=env.device)
    forces *= torch.tensor(force_std, device=env.device)
    
    torques = torch.randn(num_target_envs, num_target_bodies, 3, device=env.device)
    torques *= torch.tensor(torque_std, device=env.device)

    # constant_force = torch.tensor([0.0, 0.0, 100.0], device=env.device)
    
    # # 2. Tile this force for all target envs and bodies
    # #    Shape: (num_target_envs, num_target_bodies, 3)
    # forces = constant_force.expand(num_target_envs, num_target_bodies, 3)
    
    # # 3. Set torques to zero
    # torques = torch.zeros(num_target_envs, num_target_bodies, 3, device=env.device)

    # 4. Create the "full" tensors, initialized to zero
    # Shape: (total_num_envs, total_num_bodies, 3)
    full_forces_buffer = torch.zeros(env.num_envs, asset.num_bodies, 3, device=env.device)
    full_torques_buffer = torch.zeros(env.num_envs, asset.num_bodies, 3, device=env.device)

    # 5. Use torch.ix_ to scatter the small tensors into the full buffers
    # This selects the specific [rows (env_ids), columns (body_indexes)]
    # in the full buffer and assigns the random values.
    full_forces_buffer[env_ids[:, None], body_indexes] = forces
    full_torques_buffer[env_ids[:, None], body_indexes] = torques
    # import ipdb; ipdb.set_trace() 

    # 6. Call the correct high-level API
    # This applies forces in the BODY frame by default.
    
    asset.set_external_force_and_torque(
        forces=full_forces_buffer,
        torques=full_torques_buffer
    )
    
    # # Create zero forces/torques for all envs, then fill in for selected env_ids
    # forces = torch.zeros(env.num_envs, num_bodies, 3, device=env.device)
    # torques = torch.zeros(env.num_envs, num_bodies, 3, device=env.device)
    
    # # Only apply random forces to the specified env_ids
    # forces[env_ids] = torch.randn(len(env_ids), num_bodies, 3, device=env.device) * torch.tensor(force_std, device=env.device)
    # torques[env_ids] = torch.randn(len(env_ids), num_bodies, 3, device=env.device) * torch.tensor(torque_std, device=env.device)
    
    # import ipdb; ipdb.set_trace() 
    
    # # Apply to specific bodies (None for positions = apply at center of mass)
    # asset.root_physx_view.apply_forces_and_torques_at_position(
    #     force_data=forces,
    #     torque_data=torques,
    #     position_data=None,  # None applies at center of mass
    #     indices=body_indexes_tensor,
    #     is_global=True
    # )



def teleport_root_with_noise(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    root_pos_noise_range: tuple[float, float],  # meters for x,y displacement
    root_rot_noise_range: tuple[float, float],  # radians for rotation
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Teleport root position (x,y) and rotation with noise."""
    # extract the used quantities
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    
    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    
    # Get current root state components (following your pattern)
    root_pos = asset.data.root_pos_w.clone()
    root_ori = asset.data.root_quat_w.clone()
    root_lin_vel = asset.data.root_lin_vel_w.clone()
    root_ang_vel = asset.data.root_ang_vel_w.clone()
    
    # Sample position noise (x, y only, keep z unchanged)
    pos_noise_xy = sample_uniform(*root_pos_noise_range, (len(env_ids), 2), asset.device)
    root_pos[env_ids, :2] += pos_noise_xy  # Add noise to x, y
    
    # Sample rotation noise (yaw only for simplicity)
    rot_noise_yaw = sample_uniform(*root_rot_noise_range, (len(env_ids),), asset.device)
    
    # Convert yaw rotation to quaternion using Isaac Lab's function
    orientations_delta = quat_from_euler_xyz(
        torch.zeros_like(rot_noise_yaw),  # roll = 0
        torch.zeros_like(rot_noise_yaw),  # pitch = 0  
        rot_noise_yaw                     # yaw = noise
    )
    
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])

    # Write the new root state (following your pattern)
    asset.write_root_state_to_sim(
        torch.cat([root_pos[env_ids], root_ori[env_ids], 
                   root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1), 
        env_ids=env_ids
    )


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
