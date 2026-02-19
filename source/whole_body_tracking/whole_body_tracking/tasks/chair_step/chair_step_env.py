from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
import isaaclab.sim as sim_utils

# from isaaclab.managers import SceneEntityCfg
# import whole_body_tracking.tasks.chair_step.mdp as mdp

class ChairStepEnv(ManagerBasedRLEnv):
    """
    Custom environment for chair stepping task.
    Updates the box pose based on the motion command at each step.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _pre_physics_step(self, actions: torch.Tensor):
        super()._pre_physics_step(actions)
        
        # Check if the "motion" command exists
        if "motion" in self.command_manager.terms:
            motion_cmd = self.command_manager.terms["motion"]
            
            # Get current box/object state from the motion command
            # The command class exposes properties that handle time-stepping and env origins
            box_pos = motion_cmd.object_pos_w
            box_quat = motion_cmd.object_quat_w
            box_lin_vel = motion_cmd.object_lin_vel_w
            box_ang_vel = motion_cmd.object_ang_vel_w

            # Find the box object in the scene
            # We assume it is named "box" as per the config
            if "box" in self.scene.rigid_objects:
                box = self.scene.rigid_objects["box"]
                
                # Update box state
                # Combine position/orientation and velocities for write_root_state_to_sim
                # root_state: [pos (3), quat (4), lin_vel (3), ang_vel (3)]
                root_state = torch.cat([box_pos, box_quat, box_lin_vel, box_ang_vel], dim=-1)
                
                # Write to simulation
                box.write_root_state_to_sim(root_state)

        # Apply Assistive Forces
        # Curriculum: Linear decay from 1.0 to 0.0 over `decay_steps`
        # decay_steps = 10_000_000.0
        # decay_steps = 2000 * 24 # appx 2k iterations * 24 steps/iter = 48k steps. 

        # if "motion" in self.command_manager.terms:
            # current_step = self.common_step_counter
            # factor = max(0.0, 1.0 - current_step / decay_steps)
            
            # curriculum_factor = torch.ones(self.num_envs, device=self.device) * factor
            
            # Parameters (Can be moved to config later)
            # kp_assist = 1000.0
            # kd_assist = 50.0
            # grav_comp_assist = 0.5 # Compensate 50% gravity
            
            # mdp.curriculum.apply_assistive_forces(
                # env=self,
                # command_name="motion",
                # asset_cfg=SceneEntityCfg("robot"),
                # stiffness=kp_assist,
                # damping=kd_assist,
                # gravity_comp=grav_comp_assist,
                # curriculum_factor=curriculum_factor
            # )
    
