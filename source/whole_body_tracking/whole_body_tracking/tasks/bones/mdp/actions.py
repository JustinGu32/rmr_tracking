from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReferenceJointPositionAction(JointPositionAction):
    """Joint action term where PD target = x_ref + scale * raw_action.

    PPO outputs in ~[-1, 1], scaled by per-joint action scale to get the
    radian delta from x_ref. The pseudotarget observation and ONNX output
    use the same format as target mode: (PD_target - default_pos) / scale.
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

        # PD target = x_ref + scale * raw_action
        self._processed_actions = x_ref + self._raw_actions * self._scale
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
