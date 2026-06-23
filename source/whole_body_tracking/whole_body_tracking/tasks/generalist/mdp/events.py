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


from whole_body_tracking.tasks.generalist.mdp.commands import MotionCommand

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


def randomize_multiple_obstacles_avoiding_path(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_names: list[str],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_dist_to_path: float,
    min_dist_between_obstacles: float = 1.0,
):
    """
    Randomize multiple obstacles (cylindrical pillars) such that they do not interfere with the
    robot's reference motion path for the current episode, and do not overlap with each other.
    """
    if not asset_names:
        return

    # 1. Access Command
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    # 2. Get Full Reference Motion Path
    # motion.body_pos_w is (Total_Steps, Num_Bodies, 3)
    anchor_idx = command.motion_anchor_body_index
    raw_motion_pos = command.motion._body_pos_w[:, anchor_idx, :3] # (Total_Steps, 3)

    # 3. Determine Episode Time Slices for Each Environment
    start_steps = command.time_steps[env_ids]
    episode_length = command.steps_collect 
    max_steps = raw_motion_pos.shape[0]
    
    device = env.device
    num_envs_to_reset = len(env_ids)
    
    # Pre-fetch all obstacle assets
    obstacles = [env.scene[name] for name in asset_names]
    num_obstacles = len(obstacles)

    # Store positions for all obstacles in this batch of envs: (num_envs, num_obstacles, 3)
    final_positions = torch.zeros(num_envs_to_reset, num_obstacles, 3, device=device)
    final_positions[..., :] = 100.0 # Default far away
    final_positions[..., 2] = 1.0   # Z height

    max_iters = 100
    
    for i in range(num_envs_to_reset):
        # Identify the time slice for this specific episode
        t_start = start_steps[i]
        t_end = min(t_start + episode_length, max_steps)
        
        # Extract path (T, 2) for collision checking (ignore Z)
        if t_start < max_steps:
             path = raw_motion_pos[t_start:t_end, :2] 
        else:
             path = torch.empty(0, 2, device=device)
        
        # Place each obstacle sequentially
        for obs_idx in range(num_obstacles):
            placed = False
            for _ in range(max_iters): 
                # Sample random (x,y) candidate
                rx = torch.rand(1, device=device) * (x_range[1] - x_range[0]) + x_range[0]
                ry = torch.rand(1, device=device) * (y_range[1] - y_range[0]) + y_range[0]
                cand = torch.cat([rx, ry], dim=0) # (2,)
                
                # Shift by path start to center near robot? 
                # The original code did: cand = cand + raw_motion_pos[t_start, :2]
                cand_world = cand + raw_motion_pos[t_start, :2]

                # 1. Check distance to Path
                dist_to_path = float('inf')
                if path.shape[0] > 0:
                    dists = torch.norm(path - cand_world, dim=1)
                    dist_to_path = torch.min(dists)
                
                if dist_to_path < min_dist_to_path:
                    continue # Try again

                # 2. Check distance to previously placed obstacles
                if obs_idx > 0:
                    # Get previously placed obstacles for this env
                    # prev_pos: (obs_idx, 3)
                    # We only care about x,y. final_positions is (num_envs, num_obstacles, 3)
                    prev_positions_xy = final_positions[i, :obs_idx, :2] 
                    
                    # cand_world is (2,)
                    # Calculate distances
                    dists_to_obs = torch.norm(prev_positions_xy - cand_world, dim=1)
                    min_dist_to_obs = torch.min(dists_to_obs)
                    
                    if min_dist_to_obs < min_dist_between_obstacles:
                        continue # Try again
                
                # If we get here, it's valid
                final_positions[i, obs_idx, 0] = cand_world[0]
                final_positions[i, obs_idx, 1] = cand_world[1]
                placed = True
                break
            
            if not placed:
                # Could not place obstacle after max_iters. 
                # It stays at default (100, 100) or we could log a warning.
                pass

    # Add Env Origins to convert local positions to global world coordinates
    # env_origins: (num_envs, 3)
    env_origins = env.scene.env_origins[env_ids]
    
    # final_positions is (num_envs, num_obstacles, 3)
    # broadcase add env_origins (num_envs, 1, 3)
    global_positions = final_positions + env_origins.unsqueeze(1)

    # 5. Orientation: Identity 
    quat = torch.zeros(num_envs_to_reset, 4, device=device)
    quat[..., 0] = 1.0 

    # 6. Update Simulation State for each obstacle asset
    for obs_idx, obstacle_asset in enumerate(obstacles):
        # obstacle_asset is RigidObject or Articulation
        # We need to write state for 'env_ids'
        
        current_states = obstacle_asset.data.root_state_w[env_ids].clone() 
        
        # Set Pos
        current_states[:, :3] = global_positions[:, obs_idx, :]
        # Set Rot
        current_states[:, 3:7] = quat
        # Set Vel
        current_states[:, 7:] = 0.0 
        
        obstacle_asset.write_root_state_to_sim(current_states, env_ids=env_ids)


def rand_float(size, min_val, max_val, device):
    return (max_val - min_val) * torch.rand(size).to(device) + min_val


def _phase_to_weight_pyramid(phase, start=0.25, end=0.75):
    weight = torch.zeros_like(phase)
    mask1 = phase < start
    mask2 = phase > end
    weight[mask1] = 1/start * phase[mask1]
    weight[mask2] = 1/(1-end) * (1 - phase[mask2])
    weight[~(mask1 | mask2)] = 1.0
    return weight


def force_based_push(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_duration: tuple[int, int],
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Randomize the external forces and torques applied to the bodies.

    This function creates a set of random forces and torques sampled from the given ranges. The number of forces
    and torques is equal to the number of bodies times the number of environments. The forces and torques are
    applied to the bodies by calling ``asset.set_external_force_and_torque``. The forces and torques are only
    applied when ``asset.write_data_to_sim()`` is called in the environment.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)

    mask_update = command.force_push_counter >= command.force_update_frequency
    if command.force_config_init is False:
        command.force_duration_per_env = torch.randint(*force_duration, (command.num_envs,), dtype=torch.int, device=command.device)
        command.force_config_init = True
    mask_finish_update = command.force_push_counter == (command.force_duration_per_env + command.force_update_frequency)
    n_finish = int(mask_finish_update.sum())

    phase = ((command.force_push_counter - command.force_update_frequency) / command.force_duration_per_env).clamp_(min=0.0, max=1.0)

    # NOTE: all body force magnitude updated at once!!
    if n_finish > 0:
        command.body_force_magnitude_buf[mask_finish_update] = torch.rand(n_finish, device=command.device, dtype=command.body_force_magnitude_buf.dtype)

    if not mask_update.any():
        command.force_push_counter.add_(1)
        return

    # Reset and re-randomize config for envs that finished their force push cycle
    if n_finish > 0:
        command.force_push_counter[mask_finish_update] = 0
        command.force_duration_per_env[mask_finish_update] = torch.randint(*force_duration, (n_finish,), dtype=torch.int, device=command.device)
        updated_dirs = torch.randn(n_finish, command.num_bodies, 3, device=command.device, dtype=command.body_force_dir_buf.dtype)
        command.body_force_dir_buf[mask_finish_update] = updated_dirs / torch.norm(updated_dirs, dim=-1, keepdim=True).clamp_(min=1e-8)

    # Compute forces only for force_push bodies (smaller allocation, no zeroing needed)
    n_force_bodies = len(command.force_push_ids)
    forces = torch.zeros(command.num_envs, n_force_bodies, 3, device=command.device, dtype=command.body_force_dir_buf.dtype)
    dir_buf = command.body_force_dir_buf[:, command.force_push_ids_rel, :]
    phase_masked = phase[mask_update]
    weight = _phase_to_weight_pyramid(phase_masked).view(-1, 1, 1)
    forces[mask_update] = (
        command.body_force_magnitude_buf[mask_update].view(-1, 1, 1)
        * weight
        * dir_buf[mask_update]
        * command.max_force
    )

    asset = env.scene[asset_cfg.name]
    positions = command.robot.data.body_pos_w[:, command.force_push_ids] + math_utils.quat_apply(
        command.robot.data.body_quat_w[:, command.force_push_ids], command.force_push_body_offsets
    )
    command.last_force_applied.copy_(forces)
    command.force_push_counter.add_(1)
    torques = torch.zeros_like(forces)
    asset.set_external_force_and_torque(forces, torques, positions=positions, body_ids=command.force_push_ids, is_global=True)


def change_compliance(env: ManagerBasedEnv,
                      env_ids, command_name: str,
                      compliance_lb: tuple[float, float],
                      compliance_ub: tuple[float, float],
                      compliance_duration: tuple[int, int],
                      start_steps: int):
    command: MotionCommand = env.command_manager.get_term(command_name)
    if env.common_step_counter < start_steps:
        return
    if command.compliance_config_init == False:
        command.compliance_duration_per_env = torch.randint(*compliance_duration, (command.num_envs,),dtype=torch.int, device=command.device)
        command.compliance_ub = torch.tensor(compliance_ub, device=command.device)
        command.compliance_lb = torch.tensor(compliance_lb, device=command.device)
        command.compliance_config_init = True
    command.compliance_counter += 1
    update_mask = command.compliance_counter == command.compliance_duration_per_env
    command.eef_stiffness_buf[update_mask] = rand_float(command.eef_stiffness_buf[update_mask].shape, command.compliance_lb, command.compliance_ub, command.device)
    command.compliance_counter[update_mask] = 0
    command.compliance_duration_per_env[update_mask] = torch.randint(*compliance_duration, (sum(update_mask),), dtype=torch.int, device=command.device)
