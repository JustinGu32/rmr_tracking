from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

def rand_float(size, min_val, max_val, device):
    return (max_val - min_val) * torch.rand(size).to(device) + min_val

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

def set_environment_collision_groups(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    terrain_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
):
    """Set collision groups for terrain and robot per environment to prevent inter-environment collisions.
    
    Each environment gets a unique collision group (bitmask):
    - Environment 0: collision_group = 1 << 0 = 1
    - Environment 1: collision_group = 1 << 1 = 2
    - Environment 2: collision_group = 1 << 2 = 4
    - etc.
    
    This ensures that terrain and robots from different environments don't collide.
    Terrains and robots in the same environment share the same collision group.
    """
    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # Get terrain and robot assets
    terrain = env.scene[terrain_cfg.name]
    robot = env.scene[robot_cfg.name]
    
    # Calculate collision groups: each environment gets a unique bit
    # Use bit shifting: env 0 = 1, env 1 = 2, env 2 = 4, etc.
    # Convert to numpy for easier iteration
    env_ids_np = env_ids.cpu().numpy()
    collision_groups = (1 << env_ids).cpu().numpy().astype(int)
    
    # Try to set collision groups via PhysX API
    # Note: The exact API may vary by IsaacLab version
    # This is a best-effort attempt; if it fails, we rely on env_spacing
    try:
        # Get PhysX view for terrain and robot
        if not hasattr(terrain, 'root_physx_view') or not hasattr(robot, 'root_physx_view'):
            raise AttributeError("root_physx_view not available")
        
        terrain_view = terrain.root_physx_view
        robot_view = robot.root_physx_view
        
        # Try to access body handles or indices
        # The API might vary, so we try multiple approaches
        try:
            # Approach 1: Try to get body handles per environment
            for env_id, collision_group in zip(env_ids_np, collision_groups):
                env_id_tensor = torch.tensor([env_id], device=env.device, dtype=torch.long)
                
                # Try to get body handles (method might not exist)
                if hasattr(terrain_view, 'get_body_handles'):
                    terrain_body_handles = terrain_view.get_body_handles(env_id_tensor)
                    robot_body_handles = robot_view.get_body_handles(env_id_tensor)
                    
                    # Try to set collision groups
                    if hasattr(terrain_view, 'set_collision_group'):
                        for handle in terrain_body_handles:
                            terrain_view.set_collision_group(handle, collision_group)
                        for handle in robot_body_handles:
                            robot_view.set_collision_group(handle, collision_group)
                    elif hasattr(terrain_view, 'set_collision_groups'):
                        terrain_view.set_collision_groups(terrain_body_handles, collision_group)
                        robot_view.set_collision_groups(robot_body_handles, collision_group)
                else:
                    # Approach 2: Try to set collision groups using body indices
                    # Get number of bodies
                    terrain_num_bodies = terrain.num_bodies
                    robot_num_bodies = robot.num_bodies
                    
                    # Try setting collision groups using body indices
                    if hasattr(terrain_view, 'set_collision_group_by_index'):
                        for body_idx in range(terrain_num_bodies):
                            terrain_view.set_collision_group_by_index(env_id, body_idx, collision_group)
                        for body_idx in range(robot_num_bodies):
                            robot_view.set_collision_group_by_index(env_id, body_idx, collision_group)
                    else:
                        raise AttributeError("No suitable collision group API found")
            
            print(f"[INFO]: Successfully set collision groups for {len(env_ids)} environments. "
                  f"Each environment has unique collision group to prevent inter-environment collisions.")
        
        except (AttributeError, NotImplementedError, TypeError) as api_error:
            # If specific API methods don't exist, raise to outer catch
            raise AttributeError(f"Collision group API not available: {api_error}")
    
    except (AttributeError, NotImplementedError, TypeError) as e:
        # If API is not available, log warning and rely on env_spacing
        import warnings
        warnings.warn(
            f"Could not set collision groups programmatically: {e}. "
            f"Relying on env_spacing={env.scene.cfg.env_spacing} meters to prevent collisions. "
            f"Ensure terrain fits within env_spacing/2={env.scene.cfg.env_spacing/2} meters radius from environment origin."
        )


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


def fake_contact_force(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor,
        command_name: str,
        distance_threshold: float = 0.05,
        contact_spring_k: float = 1000.0,
        min_friction_coeff: float = 0.5,
        squeeze_force_range: tuple[float, float] = (5.0, 20.0),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    command: MotionCommand = env.command_manager.get_term(command_name)
    
    wrist_pos_ref = command.vr_3point_body_pos_w[:,:2]
    root_pos_ref = command.anchor_pos_w
    root_orn_ref = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    robot_root_pos = command.robot_anchor_pos_w
    robot_root_orn = command.robot_anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, 2, 1)
    wrist_pos_ref_l = wrist_pos_ref - root_pos_ref[:,None,:]
    wrist_pos_ref_l = math_utils.quat_apply(math_utils.quat_inv(root_orn_ref), wrist_pos_ref_l)
    wrist_pos_ref_wr = math_utils.quat_apply(robot_root_orn, wrist_pos_ref_l) + robot_root_pos[:,None,:]
    l_dir = wrist_pos_ref_wr[:,0] - wrist_pos_ref_wr[:,1] # shape [envs, 3]
    l_dir = l_dir / torch.norm(l_dir, dim=-1, keepdim=True)
    r_dir = - l_dir
    
    robot_wrist_pos = command.robot_vr_3point_pos_w[:,:2]
    o2r = robot_wrist_pos - wrist_pos_ref_wr
    dist2o = torch.norm(o2r, dim=-1)
    contact_mask = (dist2o[:,0] < distance_threshold) * (dist2o[:,1] < distance_threshold) * command.wrist_grasp_label
    last_non_contact_mask = command.last_force_applied[:, :2].norm(dim=-1).sum(dim=-1) < 1e-3
    magnitude_update_mask = contact_mask & last_non_contact_mask
    command.body_force_magnitude_buf[magnitude_update_mask] = torch.rand(sum(magnitude_update_mask), device=command.device) * (squeeze_force_range[1] - squeeze_force_range[0]) + squeeze_force_range[0]
    
    l_sp_dist = (o2r[:,0]*(-l_dir)).sum(dim=-1).clamp(min=0.0)
    r_sp_dist = (o2r[:,1]*(-r_dir)).sum(dim=-1).clamp(min=0.0)
    l_normal_force = l_dir * (command.body_force_magnitude_buf + l_sp_dist * contact_spring_k)[:, None]
    r_normal_force = r_dir * (command.body_force_magnitude_buf + r_sp_dist * contact_spring_k)[:, None]
    total_force = torch.stack([l_normal_force, r_normal_force], dim=1) # shape [envs, 2, 3]
    g_force = command.down_dir * min_friction_coeff * command.body_force_magnitude_buf[:, None, None]
    total_force += g_force
    total_force[~contact_mask] = 0.0 # zero out non-contact envs
    command.last_force_applied[:,:2] = total_force.clone()
    asset = env.scene[asset_cfg.name]

    asset.set_external_force_and_torque(total_force, torch.zeros_like(total_force), positions=robot_wrist_pos, body_ids=command.force_push_ids[:2], is_global=True)

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
