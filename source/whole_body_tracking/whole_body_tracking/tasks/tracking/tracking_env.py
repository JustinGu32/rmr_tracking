from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv


class TrackingEnv(ManagerBasedRLEnv):
    """
    Custom environment for tracking task.
    Adds ground contact force logging for feet.
    """

    def step(self, action: torch.Tensor):
        result = super().step(action)

        # Log ground reaction forces for feet
        contact = self.scene["contact_forces"]
        forces = contact.data.net_forces_w  # (num_envs, num_bodies, 3)
        body_names = contact.body_names
        left_idx = body_names.index("left_ankle_roll_link")
        right_idx = body_names.index("right_ankle_roll_link")
        self.extras["log"]["contact/left_foot_force"] = float(torch.norm(forces[:, left_idx], dim=-1).mean())
        self.extras["log"]["contact/right_foot_force"] = float(torch.norm(forces[:, right_idx], dim=-1).mean())
        self.extras["log"]["contact/total_foot_force"] = float(
            (torch.norm(forces[:, left_idx], dim=-1) + torch.norm(forces[:, right_idx], dim=-1)).mean()
        )

        return result
