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

        return super().step(action)

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
        """Apply PD spring force toward the reference motion, scaled by curriculum."""
        spring_cfg = getattr(self.cfg, "spring_force_cfg", None)
        if spring_cfg is None:
            return

        # Hard cutoff: after cutoff_steps, disable spring force permanently
        cutoff_steps = spring_cfg.get("cutoff_steps", None)
        if cutoff_steps is not None and self.common_step_counter >= cutoff_steps:
            if self._spring_force_active:
                # Clear persistent force buffer once
                self._spring_force_active = False
                self._last_spring_force = None
                robot = self.scene["robot"]
                anchor_idx = self.command_manager.get_term(spring_cfg["command_name"]).robot_anchor_body_index
                zero_force = torch.zeros(self.num_envs, 1, 3, device=self.device)
                robot.set_external_force_and_torque(
                    zero_force, zero_force, body_ids=[anchor_idx], is_global=True
                )
            return

        # Get curriculum factor: 1.0 = full assist, 0.0 = no assist
        # Quadratic scaling: (1 - difficulty_frac)^2
        # This makes the force decay faster and become truly negligible at high ADR,
        # eliminating the cliff that occurs with linear scaling + hard cutoff.
        curriculum_factor = 1.0
        curriculum_cfg = self.cfg.curriculum
        if hasattr(curriculum_cfg, "adr") and curriculum_cfg.adr is not None:
            adr_scheduler = curriculum_cfg.adr.func
            if hasattr(adr_scheduler, "difficulty_frac"):
                linear_factor = 1.0 - float(adr_scheduler.difficulty_frac)
                curriculum_factor = linear_factor ** 2  # quadratic decay

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
        super()._reset_idx(env_ids)

        # Log curriculum metrics (same pattern as other managers in _reset_idx)
        curriculum_cfg = self.cfg.curriculum
        if hasattr(curriculum_cfg, "adr") and curriculum_cfg.adr is not None:
            adr_scheduler = curriculum_cfg.adr.func
            if hasattr(adr_scheduler, "difficulty_frac"):
                self.extras["log"]["curriculum/difficulty_frac"] = float(adr_scheduler.difficulty_frac)
                self.extras["log"]["curriculum/mean_difficulty"] = float(
                    adr_scheduler.current_adr_difficulties.mean()
                )
                self.extras["log"]["curriculum/curriculum_factor"] = float(
                    1.0 - adr_scheduler.difficulty_frac
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
