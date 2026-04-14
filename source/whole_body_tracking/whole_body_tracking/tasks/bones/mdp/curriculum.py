from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import carb
import omni.physics.tensors.impl.api as physx
import isaaclab.sim as sim_utils
from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class GravityScheduler(ManagerTermBase):
    """Curriculum that linearly ramps sim gravity from a starting value to full gravity.

    Starts at start_gravity and linearly interpolates to end_gravity over ramp_steps,
    beginning after start_steps.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.difficulty_frac = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_gravity: tuple[float, float, float] = (0.0, 0.0, -2.0),
        end_gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
        start_steps: int = 0,
        ramp_steps: int = 5000,
    ):
        if env.common_step_counter <= start_steps:
            self.difficulty_frac = 0.0
        else:
            t = (env.common_step_counter - start_steps) / max(ramp_steps - start_steps, 1)
            self.difficulty_frac = min(float(t), 1.0)

        # Interpolate gravity
        gx = start_gravity[0] + self.difficulty_frac * (end_gravity[0] - start_gravity[0])
        gy = start_gravity[1] + self.difficulty_frac * (end_gravity[1] - start_gravity[1])
        gz = start_gravity[2] + self.difficulty_frac * (end_gravity[2] - start_gravity[2])

        # Apply to physics scene
        physics_sim_view: physx.SimulationView = sim_utils.SimulationContext.instance().physics_sim_view
        physics_sim_view.set_gravity(carb.Float3(gx, gy, gz))

        return self.difficulty_frac
