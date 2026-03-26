from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReferenceJointPositionAction(JointPositionAction):
    """Joint action term where PD target = x_ref + raw_action.

    The PPO output directly represents the delta (x_target - x_ref) in radians.
    No action scale is applied to the delta.
    """

    def __init__(self, cfg: ReferenceJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._command_name = cfg.command_name

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        # get reference joint positions from the motion command
        motion_command = self._env.command_manager.get_term(self._command_name)
        x_ref = motion_command.joint_pos
        if not isinstance(self._joint_ids, slice):
            x_ref = x_ref[:, self._joint_ids]

        # Clip raw delta to prevent simulation blowup from large initial std
        delta = self._raw_actions
        if self.cfg.delta_clip is not None:
            delta = torch.clamp(delta, min=-self.cfg.delta_clip, max=self.cfg.delta_clip)

        # PD target = x_ref + delta (no scale — PPO output is the actual radian delta)
        self._processed_actions = x_ref + delta
        # clip processed actions (absolute joint positions)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )


@configclass
class ReferenceJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for reference-tracking joint position action."""

    class_type: type = ReferenceJointPositionAction
    command_name: str = "motion"
    """Name of the MotionCommand term in the command manager."""

    delta_clip: float | None = 0.5
    """Max absolute delta in radians. Clips raw PPO output before adding to x_ref.
    Prevents simulation blowup from large initial exploration noise. Default: 0.5 rad (~28 deg)."""
