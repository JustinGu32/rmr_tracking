from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

import whole_body_tracking.tasks.staircase.mdp as mdp


class StaircaseEnv(ManagerBasedRLEnv):
    """
    Custom environment for staircase task.
    Updates the staircase pose based on the motion command at each step.
    Assistive spring forces are applied via the event manager (EventCfg.assistive_spring_force)
    and scheduled by the ADR curriculum (CurriculumCfg.adr + spring_force_adr).
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # These attrs are set by the apply_assistive_spring_force event function
        self._last_spring_force = None
        self._spring_force_active = True
        self._spring_force_curriculum_factor = 1.0

    def step(self, action: torch.Tensor):
        self._update_staircase_pose()
        # return super().step(action)
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

    # --- Linear-ramp _apply_spring_force removed ---
    # Spring force is now applied by EventCfg.assistive_spring_force (interval event)
    # and curriculum_factor is controlled by CurriculumCfg.spring_force_adr (ADR-based).

    def _update_staircase_pose(self):
        """Sync staircase rigid body pose with the motion command's object state."""
        motion_cmd = self.command_manager.get_term("motion")

        box_pos = motion_cmd.object_pos_w
        box_quat = motion_cmd.object_quat_w
        box_lin_vel = motion_cmd.object_lin_vel_w
        box_ang_vel = motion_cmd.object_ang_vel_w

        if "staircase" in self.scene.rigid_objects:
            staircase = self.scene.rigid_objects["staircase"]
            root_state = torch.cat([box_pos, box_quat, box_lin_vel, box_ang_vel], dim=-1)
            staircase.write_root_state_to_sim(root_state)

    def _reset_idx(self, env_ids: Sequence[int]):
        # # Debug: print which terminations fired
        # for i, term_name in enumerate(self.termination_manager._term_names):
        #     fired = self.termination_manager._term_dones[env_ids, i]
        #     if fired.any():
        #         count = fired.sum().item()
        #         print(f"  Termination '{term_name}' fired in {int(count)}/{len(env_ids)} envs")

        super()._reset_idx(env_ids)

        # Log ADR curriculum metrics
        curriculum_cfg = self.cfg.curriculum
        if hasattr(curriculum_cfg, "adr") and curriculum_cfg.adr is not None:
            adr_scheduler = curriculum_cfg.adr.func
            if hasattr(adr_scheduler, "difficulty_frac"):
                self.extras["log"]["curriculum/difficulty_frac"] = float(adr_scheduler.difficulty_frac)
                self.extras["log"]["curriculum/mean_difficulty"] = float(
                    adr_scheduler.current_adr_difficulties.mean()
                )

        # Log curriculum_factor (set by the event function)
        curriculum_factor = getattr(self, "_spring_force_curriculum_factor", 1.0)
        self.extras["log"]["curriculum/curriculum_factor"] = float(curriculum_factor)

        # Log spring force info (set by the event function)
        spring_force_active = getattr(self, "_spring_force_active", True)
        self.extras["log"]["curriculum/spring_force_active"] = float(spring_force_active)
        last_force = getattr(self, "_last_spring_force", None)
        if last_force is not None:
            force_mag = torch.norm(last_force, dim=-1).mean()
            self.extras["log"]["curriculum/spring_force_magnitude"] = float(force_mag)
            # Per-axis: XY (horizontal) and Z (vertical)
            force_xy = torch.norm(last_force[:, :2], dim=-1).mean()
            force_z = last_force[:, 2].abs().mean()
            self.extras["log"]["curriculum/spring_force_xy"] = float(force_xy)
            self.extras["log"]["curriculum/spring_force_z"] = float(force_z)

