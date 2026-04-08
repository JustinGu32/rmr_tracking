import os

from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.staircase.mdp as mdp
from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.staircase.staircase_env_cfg import StaircaseEnvCfg, CurriculumCfg
from whole_body_tracking.tasks.staircase.staircase_compliance_cfg import StaircaseComplianceCfg

@configclass
class G1StaircaseEnvCfg(StaircaseEnvCfg):
    """G1 robot configuration for staircase."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
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

        assist_mode = os.environ.get("WBT_ASSIST_MODE", "both")
        if assist_mode not in {"both", "gravity_only", "spring_only", "none"}:
            raise ValueError(f"Unsupported WBT_ASSIST_MODE: {assist_mode}")

        if assist_mode == "gravity_only":
            self.events.assistive_spring_force.params["stiffness"] = 0.0
            self.events.assistive_spring_force.params["damping"] = 0.0
            self.events.assistive_spring_force.params["ang_stiffness"] = 0.0
        elif assist_mode == "spring_only":
            self.events.assistive_spring_force.params["gravity_comp"] = 0.0
        elif assist_mode == "none":
            self.events.assistive_spring_force.params["stiffness"] = 0.0
            self.events.assistive_spring_force.params["damping"] = 0.0
            self.events.assistive_spring_force.params["ang_stiffness"] = 0.0
            self.events.assistive_spring_force.params["gravity_comp"] = 0.0

        if os.environ.get("WBT_DOUBLE_STEP") == "1":
            self.rewards.double_step_penalty = RewTerm(
                func=mdp.double_step_penalty,
                weight=0.5,
                params={
                    "command_name": "motion",
                    "threshold": 2.0,
                    "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                },
            )

        if os.environ.get("WBT_MOTION_JOINT_POS") == "1":
            self.rewards.motion_joint_pos = RewTerm(
                func=mdp.motion_joint_position_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.15},
            )

        if os.environ.get("WBT_CURRICULUM") == "1":
            self.curriculum = CurriculumCfg()

@configclass
class G1StaircasePlayCfg(G1StaircaseEnvCfg):
    """G1 robot configuration for staircase playback (no perturbations)."""
    def __post_init__(self):
        super().__post_init__()
        # Disable perturbations for clean playback (unless --push flag is set)
        if os.environ.get("WBT_PUSH") != "1":
            self.events.push_robot = None
        # Disable spring force and curriculum
        self.events.assistive_spring_force = None
        if self.curriculum is not None:
            self.curriculum.spring_force_linear = None
            self.curriculum.spring_force_factor = None
        self.episode_length_s = 25.0
        self.commands.motion.min_sample_idx = 0
        self.commands.motion.max_sample_idx = 0
        self.terminations.anchor_pos_xy = None


@configclass
class G1StaircaseComplianceCfg(StaircaseComplianceCfg):
    """G1 robot configuration for staircase with CHIP compliance."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
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

        # CHIP: Add history to proprioception (optional but recommended)
        self.observations.policy.gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=4)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=4)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=4)
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=4)
        self.observations.policy.actions = ObsTerm(func=mdp.last_action, history_length=4)
        self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        self.observations.critic.actions = ObsTerm(func=mdp.last_action, history_length=10)
        
        # CHIP: VR position tracking reward
        self.rewards.vr_position = RewTerm(
            func=mdp.vr_position_relative_error_exp,
            weight=2.0,
            params={"command_name": "motion", "std": 0.1},
        )



        if os.environ.get("WBT_DOUBLE_STEP") == "1":
            self.rewards.double_step_penalty = RewTerm(
                func=mdp.double_step_penalty,
                weight=0.5,
                params={
                    "command_name": "motion",
                    "threshold": 2.0,
                    "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                },
            )

        if os.environ.get("WBT_MOTION_JOINT_POS") == "1":
            self.rewards.motion_joint_pos = RewTerm(
                func=mdp.motion_joint_position_error_exp,
                weight=1.0,
                params={"command_name": "motion", "std": 0.15},
            )

@configclass
class G1StaircaseCompliancePlayCfg(G1StaircaseComplianceCfg):
    """G1 robot configuration for staircase with CHIP compliance."""

    def __post_init__(self):
        super().__post_init__()
        if os.environ.get("WBT_PUSH") != "1":
            self.events.push_robot = None
        if os.environ.get("WBT_PUSH_FEET") != "1":
            self.events.force_push_robot = None
        self.terminations.anchor_pos_xy = None
        self.episode_length_s = 25.0
        self.commands.motion.min_sample_idx = 0
        self.commands.motion.max_sample_idx = 0
        self.events.assistive_spring_force = None

        # self.events.change_compliance = None
        self.events.change_compliance = EventTerm(
            func=mdp.change_compliance,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "command_name": "motion",
                "compliance_lb": [0.00, 0.00, 0.00],
                "compliance_ub": [0.09, 0.09, 0.09],
                "compliance_duration": (1, 2),
                "start_steps": 0,
            }
        )
        


        


@configclass
class G1StaircasePlayEnvCfg(StaircaseEnvCfg):
    """G1 robot configuration for staircase."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = G1_ACTION_SCALE
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
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

        self.events.assistive_spring_force = None
        if self.curriculum is not None:
            self.curriculum.spring_force_linear = None
            self.curriculum.spring_force_factor = None
        self.episode_length_s = 25.0
        self.commands.motion.min_sample_idx = 0
        self.commands.motion.max_sample_idx = 0
        self.terminations.anchor_pos_xy = None
