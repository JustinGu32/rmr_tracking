from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class StaircaseEnv(ManagerBasedRLEnv):
    """Tracking environment with a static staircase object.

    Identical to the BeyondMimic ``tracking`` task except that a kinematic staircase
    rigid object is placed in the scene. The staircase pose is synced to the motion
    command's (static, per-env) object pose each step.
    """

    def step(self, action: torch.Tensor):
        self._update_staircase_pose()
        return super().step(action)

    def _update_staircase_pose(self):
        """Sync the staircase rigid body pose with the motion command's object state."""
        if "staircase" not in self.scene.rigid_objects:
            return
        motion_cmd = self.command_manager.get_term("motion")
        root_state = torch.cat(
            [
                motion_cmd.object_pos_w,
                motion_cmd.object_quat_w,
                motion_cmd.object_lin_vel_w,
                motion_cmd.object_ang_vel_w,
            ],
            dim=-1,
        )
        self.scene.rigid_objects["staircase"].write_root_state_to_sim(root_state)
