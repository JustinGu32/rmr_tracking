import os

from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.bones.config.g1.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.bones.bones_env_cfg import BonesEnvCfg, ROUGH_TERRAINS_CFG
from whole_body_tracking.tasks.bones.bones_env_cfg_3pt import Bones3ptEnvCfg, SpringForceCurriculumCfg
from whole_body_tracking.tasks.bones.bones_env_cfg_3pt_binded import Bones3ptBindedEnvCfg
from whole_body_tracking.tasks.bones.bones_env_cfg_3pt_terrain import Bones3ptTerrainEnvCfg
from whole_body_tracking.tasks.bones.bones_env_cfg_3pt_multi_terrain import Bones3ptMultiTerrainEnvCfg
from whole_body_tracking.tasks.bones.bones_env_cfg_3pt_binded_multi_terrain import Bones3ptBindedMultiTerrainEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.terrains import TerrainImporterCfg
import isaaclab.sim as sim_utils
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
import whole_body_tracking.tasks.bones.mdp as mdp

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


def _g1_post_init(self):
    """Common G1 post-init: set robot, action scale, anchor, and body names."""
    self.scene.robot = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    self.actions.joint_pos.scale = G1_ACTION_SCALE
    self.commands.motion.anchor_body_name = "pelvis"
    self.commands.motion.body_names = G1_BODY_NAMES


@configclass
class G1BonesEnvCfg(BonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)

@configclass
class G1BonesRoughEnvCfg(BonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)
        self.scene.terrain = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=ROUGH_TERRAINS_CFG,
                max_init_terrain_level=10,
                collision_group=1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                ),
                visual_material=sim_utils.MdlFileCfg(
                    mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                    project_uvw=True,
                ),
                debug_vis=False,
            )

@configclass
class G1BonesGraspEnvCfg(BonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)
        self.commands.motion.has_wrist_grasp_label = True

        self.events.force_push_robot = EventTerm(
            func=mdp.fake_contact_force,
            mode="interval",
            interval_range_s = (0.02, 0.02),
            params={
                "distance_threshold": 0.05,
                "contact_spring_k": 1000.0,
                "min_friction_coeff": 0.5,
                "squeeze_force_range": (5.0, 20.0),
                "command_name": "motion",
                "asset_cfg": mdp.SceneEntityCfg("robot", joint_names=[".*"]),
            }
        )
        self.observations.policy.grasp_label = ObsTerm(func=mdp.wrist_grasp_label, params={"command_name": "motion"})
        self.observations.critic.grasp_label = ObsTerm(func=mdp.wrist_grasp_label, params={"command_name": "motion"})
        self.rewards.grasp_contact = RewTerm(func=mdp.grasp_contact_reward, weight=1.0, params={
            "command_name": "motion",
            "distance_threshold": 0.05,
        })

@configclass
class G1Bones3ptEnvCfg(Bones3ptEnvCfg):
    spring_force_cfg: dict = {
        "command_name": "motion",
        "body_names": ["torso_link"],
        "stiffness": 2000.0,
        "ang_stiffness": 300.0,
        "damping": 15.0,
        "axis_weights": [0.5, 0.5, 2.0],
    }

    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)



@configclass
class G1Bones3ptBindedEnvCfg(Bones3ptBindedEnvCfg):
    """G1 3pt tracking with binded motion (each env bound to motion via env_id % num_motions)."""

    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)


@configclass
class G1Bones3ptComplianceEnvCfg(Bones3ptEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)
        self.observations.policy.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.policy.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos_compliant = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})

        # For all proprioception add 10 step history (matching CHIP paper)
        self.observations.policy.gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=10)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10)
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=10)
        action_obs_func = mdp.last_action_pseudotarget if os.environ.get("BONES_PPO_OUTPUT") == "delta-pseudotarget" else mdp.last_action
        self.observations.policy.actions = ObsTerm(func=action_obs_func, history_length=10)

        self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        self.observations.critic.actions = ObsTerm(func=action_obs_func, history_length=10)

        self.events.change_compliance = EventTerm(func=mdp.change_compliance,
                                                  mode="interval",
                                                  interval_range_s=(0.02, 0.02),
                                                  params={
                                                        "command_name": "motion",
                                                        "compliance_lb": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                        # "compliance_ub": [0.09, 0.09, 0.0, 0.09, 0.09, 0.09],
                                                        "compliance_ub": [0.05, 0.05, 0.0, 0.05, 0.05, 0.05],
                                                        "compliance_duration": (100, 200),
                                                        "start_steps": 0
                                                  })
        self.rewards.vr_position_upper = RewTerm(func=mdp.vr_position_relative_error_exp,
                                                 weight=2.0,
                                                 params={
                                                        "command_name": "motion",
                                                        "std": 0.1,
                                                        "body_indices": [0, 1, 2],  # wrists + torso
                                                 })
        self.rewards.vr_position_lower = RewTerm(func=mdp.vr_position_relative_error_exp,
                                                 weight=2.0,
                                                 params={
                                                        "command_name": "motion",
                                                        "std": 0.1,
                                                        "body_indices": [3, 4, 5],  # ankles + pelvis
                                                 })

@configclass
class G1Bones3ptCompliancePlayCfg(G1Bones3ptComplianceEnvCfg):
    """G1 3pt compliance play config: start at t=0, full episode, no push."""

    def __post_init__(self):
        super().__post_init__()
        # Disable force perturbations
        self.events.force_push_robot = None
        self.events.push_robot = None
        self.events.change_compliance = EventTerm(func=mdp.change_compliance,
                                                  mode="interval",
                                                  interval_range_s=(0.02, 0.02),
                                                  params={
                                                        "command_name": "motion",
                                                        "compliance_lb": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                        "compliance_ub": [0.05, 0.05, 0.0, 0.05, 0.05, 0.05],
                                                        "compliance_duration": (1, 2),
                                                        "start_steps": 0
                                                  })
        # Start at timestep 0, run full episode
        self.commands.motion.min_sample_idx = 0
        self.commands.motion.max_sample_idx = 0
        self.episode_length_s = 25.0

@configclass
class G1Bones3ptBindedComplianceEnvCfg(Bones3ptBindedEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)
        self.observations.policy.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.policy.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos_compliant = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})

        # For all proprioception add 10 step history (matching CHIP paper)
        self.observations.policy.gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=10)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10)
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=10)
        action_obs_func = mdp.last_action_pseudotarget if os.environ.get("BONES_PPO_OUTPUT") == "delta-pseudotarget" else mdp.last_action
        self.observations.policy.actions = ObsTerm(func=action_obs_func, history_length=10)

        self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        self.observations.critic.actions = ObsTerm(func=action_obs_func, history_length=10)

        self.events.change_compliance = EventTerm(func=mdp.change_compliance,
                                                  mode="interval",
                                                  interval_range_s=(0.02, 0.02),
                                                  params={
                                                        "command_name": "motion",
                                                        "compliance_lb": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                        "compliance_ub": [0.05, 0.05, 0.0, 0.05, 0.05, 0.0],
                                                        "compliance_duration": (100, 200),
                                                        "start_steps": 0
                                                  })

@configclass
class G1BonesWoStateEstimationEnvCfg(G1BonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class G1BonesLowFreqEnvCfg(G1BonesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE

@configclass
class G1Bones3ptTerrainEnvCfg(Bones3ptTerrainEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)

@configclass
class G1Bones3ptMultiTerrainEnvCfg(Bones3ptMultiTerrainEnvCfg):
    terrain_folder_path: str = "lafan_terrain/merged"

    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)

@configclass
class G1Bones3ptBindedMultiTerrainEnvCfg(Bones3ptBindedMultiTerrainEnvCfg):
    """G1 3pt tracking with binded motion-terrain pairs (each env gets motion and terrain with same base name).

    Requires --motion_dir and --terrain_folder (or --terrain_folder for terrain) when running train.py.
    """

    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)


@configclass
class G1BonesMultiClipEnvCfg(Bones3ptEnvCfg):
    """G1 bones multi-clip training from a Zarr store."""

    def __post_init__(self):
        super().__post_init__()
        _g1_post_init(self)
        self.commands.motion = mdp.ZarrMultiMotionCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e9, 1.0e9),
            adaptive_uniform_ratio=0.25,
            debug_vis=True,
            pose_range={
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (-0.01, 0.01),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.2, 0.2),
            },
            velocity_range={
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.2, 0.2),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            },
            joint_position_range=(-0.1, 0.1),
        )
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = G1_BODY_NAMES


@configclass
class G1BonesMultiClipComplianceEnvCfg(G1BonesMultiClipEnvCfg):
    """G1 bones multi-clip with CHIP compliance."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.policy.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos_compliant = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        self.observations.critic.vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})

        # For all proprioception add 10 step history (matching CHIP paper)
        self.observations.policy.gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10)
        self.observations.policy.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2), history_length=10)
        self.observations.policy.joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=10)
        self.observations.policy.joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=10)
        action_obs_func = mdp.last_action_pseudotarget if os.environ.get("BONES_PPO_OUTPUT") == "delta-pseudotarget" else mdp.last_action
        self.observations.policy.actions = ObsTerm(func=action_obs_func, history_length=10)

        self.observations.critic.base_lin_vel = ObsTerm(func=mdp.base_lin_vel, history_length=10)
        self.observations.critic.base_ang_vel = ObsTerm(func=mdp.base_ang_vel, history_length=10)
        self.observations.critic.joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=10)
        self.observations.critic.joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=10)
        self.observations.critic.actions = ObsTerm(func=action_obs_func, history_length=10)

        self.events.change_compliance = EventTerm(func=mdp.change_compliance,
                                                  mode="interval",
                                                  interval_range_s=(0.02, 0.02),
                                                  params={
                                                        "command_name": "motion",
                                                        "compliance_lb": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                                        "compliance_ub": [0.05, 0.05, 0.0, 0.05, 0.05, 0.05],
                                                        "compliance_duration": (100, 200),
                                                        "start_steps": 0
                                                  })
        self.rewards.vr_position_upper = RewTerm(func=mdp.vr_position_relative_error_exp,
                                                 weight=2.0,
                                                 params={
                                                        "command_name": "motion",
                                                        "std": 0.1,
                                                        "body_indices": [0, 1, 2],  # wrists + torso
                                                 })
        self.rewards.vr_position_lower = RewTerm(func=mdp.vr_position_relative_error_exp,
                                                 weight=2.0,
                                                 params={
                                                        "command_name": "motion",
                                                        "std": 0.1,
                                                        "body_indices": [3, 4, 5],  # ankles + pelvis
                                                 })
