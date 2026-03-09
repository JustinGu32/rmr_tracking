from __future__ import annotations

import torch
from collections.abc import Sequence

from isaaclab.envs import ManagerBasedRLEnv

import whole_body_tracking.tasks.staircase.mdp as mdp


class StaircaseEnv(ManagerBasedRLEnv):
    """
    Custom environment for staircase task.
    Updates the staircase pose based on the motion command at each step
    and applies spring-based assistive forces toward the reference motion.
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._last_spring_force = None
        self._spring_force_active = True

    def step(self, action: torch.Tensor):
        # --- Apply spring force BEFORE physics stepping ---
        # We set the force buffer here; it gets flushed to simulation
        # inside super().step() via scene.write_data_to_sim() each substep.
        self._apply_spring_force()
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

    def _apply_spring_force(self):
        """Apply PD spring force toward the reference motion, scaled by linear ramp."""
        spring_cfg = getattr(self.cfg, "spring_force_cfg", None)
        if spring_cfg is None:
            return

        # Linear ramp: curriculum_factor goes from 1.0 → 0.0 over ramp_steps
        ramp_steps = spring_cfg.get("ramp_steps", 480000)  # default: 20k iters × 24
        t = min(self.common_step_counter / ramp_steps, 1.0)
        curriculum_factor = 1.0 - t  # linear decay, no discontinuity

        if curriculum_factor <= 0.0:
            # Force fully ramped down — clear any persistent force buffer
            if self._spring_force_active:
                self._spring_force_active = False
                self._last_spring_force = None
                robot = self.scene["robot"]
                anchor_idx = self.command_manager.get_term(spring_cfg["command_name"]).robot_anchor_body_index
                zero_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
                robot.set_external_force_and_torque(
                    zero_force, zero_force, body_ids=[anchor_idx], is_global=True
                )
            return

        self._last_spring_force = mdp.apply_spring_force(
            env=self,
            command_name=spring_cfg["command_name"],
            asset_name="robot",
            stiffness=spring_cfg["stiffness"],
            ang_stiffness=spring_cfg.get("ang_stiffness", 100.0),
            damping=spring_cfg["damping"],
            axis_weights=tuple(spring_cfg["axis_weights"]),
            curriculum_factor=curriculum_factor,
        )


    def _reset_idx(self, env_ids: Sequence[int]):
        # # Debug: print which terminations fired
        # for i, term_name in enumerate(self.termination_manager._term_names):
        #     fired = self.termination_manager._term_dones[env_ids, i]
        #     if fired.any():
        #         count = fired.sum().item()
        #         print(f"  Termination '{term_name}' fired in {int(count)}/{len(env_ids)} envs")

        super()._reset_idx(env_ids)

        # Log linear ramp curriculum metrics
        spring_cfg = getattr(self.cfg, "spring_force_cfg", None)
        if spring_cfg is not None:
            ramp_steps = spring_cfg.get("ramp_steps", 480000)
            t = min(self.common_step_counter / ramp_steps, 1.0)
            curriculum_factor = 1.0 - t
            self.extras["log"]["curriculum/curriculum_factor"] = float(curriculum_factor)
            self.extras["log"]["curriculum/ramp_progress"] = float(t)

        # Log ADR metrics if still active (informational only)
        curriculum_cfg = self.cfg.curriculum
        if hasattr(curriculum_cfg, "adr") and curriculum_cfg.adr is not None:
            adr_scheduler = curriculum_cfg.adr.func
            if hasattr(adr_scheduler, "difficulty_frac"):
                self.extras["log"]["curriculum/difficulty_frac"] = float(adr_scheduler.difficulty_frac)
                self.extras["log"]["curriculum/mean_difficulty"] = float(
                    adr_scheduler.current_adr_difficulties.mean()
                )

        # Log spring force info
        self.extras["log"]["curriculum/spring_force_active"] = float(self._spring_force_active)
        if self._last_spring_force is not None:
            force_mag = torch.norm(self._last_spring_force, dim=-1).mean()
            self.extras["log"]["curriculum/spring_force_magnitude"] = float(force_mag)
            # Per-axis: XY (horizontal) and Z (vertical)
            force_xy = torch.norm(self._last_spring_force[:, :2], dim=-1).mean()
            force_z = self._last_spring_force[:, 2].abs().mean()
            self.extras["log"]["curriculum/spring_force_xy"] = float(force_xy)
            self.extras["log"]["curriculum/spring_force_z"] = float(force_z)
