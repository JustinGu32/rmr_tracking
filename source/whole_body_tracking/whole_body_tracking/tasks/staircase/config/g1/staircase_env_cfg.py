import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.staircase.mdp as mdp
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.staircase.staircase_env_cfg import StaircaseEnvCfg

G1_BODY_NAMES = [
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]


@configclass
class G1StaircaseEnvCfg(StaircaseEnvCfg):
    """G1 robot configuration for the staircase tracking task."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = G1_BODY_NAMES


@configclass
class G1StaircaseObsAugEnvCfg(G1StaircaseEnvCfg):
    """Obs-space experiment (Run B): adds global tracking info to the POLICY obs.

    On top of the default policy obs (tracked bodies = cfg.body_names, the 14 G1 links):
      - body_pos_env : tracked-body positions relative to the env origin (translation-invariant)
      - body_lin_vel : tracked-body world-frame linear velocity
      - time_left    : normalized fraction of the motion clip remaining
    Everything else is identical to G1StaircaseEnvCfg.
    """

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.body_pos_env = ObsTerm(
            func=mdp.robot_body_pos_env, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        self.observations.policy.body_lin_vel = ObsTerm(
            func=mdp.robot_body_lin_vel_tracked, params={"command_name": "motion"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        )
        self.observations.policy.time_left = ObsTerm(
            func=mdp.time_left, params={"command_name": "motion"}
        )


@configclass
class G1StaircasePlayEnvCfg(G1StaircaseEnvCfg):
    """G1 staircase playback config (no perturbations)."""

    def __post_init__(self):
        super().__post_init__()
        if os.environ.get("WBT_PUSH") != "1":
            self.events.push_robot = None
