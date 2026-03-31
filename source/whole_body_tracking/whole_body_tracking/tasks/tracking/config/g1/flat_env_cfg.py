from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg
from whole_body_tracking.tasks.tracking.tracking_collect_cfg import TrackingCollectCfg
from whole_body_tracking.tasks.tracking.tracking_sim_cfg import TrackingSimCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.mdp.commands import MultiClipMotionCommandCfg


@configclass
class G1FlatEnvCfg(TrackingEnvCfg):
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

        self.terminations.bad_anchor_pos_xy = DoneTerm(
            func=mdp.bad_anchor_pos_x_y_only,
            params={"command_name": "motion", "threshold": 0.7},
        )


@configclass
class G1FlatPlayEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.commands.motion.min_sample_idx = 0
        self.commands.motion.max_sample_idx = 0
        self.curriculum.adr = None
        self.curriculum.spring_force_adr = None
        self.spring_force_cfg = None
        self.events.push_robot = None
        self.events.force_push_robot = None
        self.terminations.bad_anchor_pos_xy = None

@configclass
class G1FlatWoStateEstimationEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1FlatLowFreqEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class G1FlatCollectEnvCfg(TrackingCollectCfg):
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



@configclass
class G1FlatSimEnvCfg(TrackingSimCfg):
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


@configclass
class G1FlatComplianceEnvCfg(G1FlatEnvCfg):
    """G1 flat tracking with CHIP feet compliance."""
    def __post_init__(self):
        super().__post_init__()

        # Compliance observations
        self.observations.policy.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.policy.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos_compliant = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})

        # Proprioception with history
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

        # CHIP compliance event
        self.events.change_compliance = EventTerm(func=mdp.change_compliance,
                                                  mode="interval",
                                                  interval_range_s=(0.02, 0.02),
                                                  params={
                                                        "command_name": "motion",
                                                        "compliance_lb": [0.0, 0.0, 0.0],
                                                        "compliance_ub": [0.05, 0.05, 0.05],
                                                        "compliance_duration": (100, 200),
                                                        "start_steps": 0
                                                  })

        # Termination
        self.terminations.bad_anchor_pos_xy = DoneTerm(
            func=mdp.bad_anchor_pos_x_y_only,
            params={"command_name": "motion", "threshold": 0.7},
        )

@configclass
class G1FlatCompliancePlayEnvCfg(G1FlatComplianceEnvCfg):
    """G1 flat tracking with CHIP feet compliance."""
    def __post_init__(self):
        super().__post_init__()
        self.terminations.bad_anchor_pos_xy = None
        self.episode_length_s = 60.0
        self.commands.motion.min_sample_idx = 0
        self.commands.motion.max_sample_idx = 0
        self.curriculum.adr = None
        self.curriculum.spring_force_adr = None
        self.spring_force_cfg = None
        self.events.change_compliance = EventTerm(
            func=mdp.change_compliance,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "command_name": "motion",
                "compliance_lb": [0.05, 0.05, 0.05],
                "compliance_ub": [0.05, 0.05, 0.05],
                "compliance_duration": (1, 2),
                "start_steps": 0,
            }
        )
        self.events.push_robot = None
        self.events.force_push_robot = None


@configclass
class G1FlatMultiClipEnvCfg(G1FlatEnvCfg):
    """G1 flat tracking with multi-clip Zarr motion loading."""
    def __post_init__(self):
        super().__post_init__()

        # Grab all the existing command settings from the parent
        parent_cfg = self.commands.motion

        # Replace with multi-clip version, preserving all parent settings
        # Disable diffusion_collect obs group — not used during RL training
        self.observations.diffusion_collect = None

        self.commands.motion = MultiClipMotionCommandCfg(
            asset_name=parent_cfg.asset_name,
            resampling_time_range=parent_cfg.resampling_time_range,
            debug_vis=parent_cfg.debug_vis,
            zarr_path="",  # Placeholder — set at runtime by train.py via --zarr_path
            anchor_body_name=parent_cfg.anchor_body_name,
            body_names=parent_cfg.body_names,
            pose_range=parent_cfg.pose_range,
            velocity_range=parent_cfg.velocity_range,
            joint_position_range=parent_cfg.joint_position_range,
            adaptive_kernel_size=parent_cfg.adaptive_kernel_size,
            adaptive_lambda=parent_cfg.adaptive_lambda,
            adaptive_uniform_ratio=parent_cfg.adaptive_uniform_ratio,
            adaptive_alpha=parent_cfg.adaptive_alpha,
            min_sample_idx=parent_cfg.min_sample_idx,
            max_sample_idx=parent_cfg.max_sample_idx,
            steps_collect=parent_cfg.steps_collect,
            force_update_frequency=parent_cfg.force_update_frequency,
            max_force=parent_cfg.max_force,
            force_push_body=parent_cfg.force_push_body,
            force_push_body_offset=parent_cfg.force_push_body_offset,
            vr_3point_body=parent_cfg.vr_3point_body,
            vr_3point_body_offset=parent_cfg.vr_3point_body_offset,
        )


@configclass
class G1FlatMultiClipPlayEnvCfg(G1FlatMultiClipEnvCfg):
    """Play-mode config for multi-clip: disables terminations, events, curriculum."""
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        self.curriculum.adr = None
        self.curriculum.spring_force_adr = None
        self.spring_force_cfg = None
        self.events.push_robot = None
        self.events.force_push_robot = None
        self.terminations.bad_anchor_pos_xy = None


