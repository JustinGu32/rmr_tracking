from __future__ import annotations

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.bones.mdp as mdp

##
# Scene definition
##

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

VELOCITY_RANGE_SOFT = {
    "x": (-0.25, 0.25),
    "y": (-0.25, 0.25),
    "z": (-0.1, 0.1),
    "roll": (-0.26, 0.26),
    "pitch": (-0.26, 0.26),
    "yaw": (-0.39, 0.39),
}


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
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
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # motion = mdp.MotionCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(1.0e9, 1.0e9),
    #     debug_vis=True,
    #     pose_range={
    #         "x": (-0.05, 0.05),
    #         "y": (-0.05, 0.05),
    #         "z": (-0.01, 0.01),
    #         "roll": (-0.1, 0.1),
    #         "pitch": (-0.1, 0.1),
    #         "yaw": (-0.2, 0.2),
    #     },
    #     velocity_range=VELOCITY_RANGE,
    #     joint_position_range=(-0.1, 0.1),
    # )
    motion = mdp.MultiMotionCommandCfg(
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
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], use_default_offset=True)
    # joint_pos = mdp.ReferenceJointPositionActionCfg(asset_name="robot", joint_names=[".*"], command_name="motion")


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        # command = ObsTerm(func=mdp.command_lower_body, params={"command_name":"motion"})
        vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name":"motion"})
        vr_3point_orn = ObsTerm(func=mdp.vr_3point_local_orn_target, params={"command_name":"motion"})
        #command = ObsTerm(func=mdp.command_lookahead, params={"command_name":"motion"})
        # motion_anchor_pos_b = ObsTerm(
        #     func=mdp.motion_anchor_pos_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.2, n_max=0.2)
        # )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-1.0, n_max=1.0))
        gravity_dir = ObsTerm(func=mdp.projected_gravity, params={"command_name": "motion"}, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PrivilegedCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # interval
    # push_robot = EventTerm(
    #     func=mdp.push_by_setting_velocity,
    #     mode="interval",
    #     interval_range_s=(1.0, 3.0),
    #     params={"velocity_range": VELOCITY_RANGE},
    # )

    # force push robot
    force_push_robot = EventTerm(
        func=mdp.force_based_push,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={"force_duration": [50,100],
                "command_name": "motion",
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # Spring force is applied intervally (every step) and modulated by the curriculum
    assistive_spring_force = EventTerm(
        func=mdp.apply_spring_force,
        mode="interval",
        interval_range_s=(0.005, 0.005),  # Assuming 200Hz control frequency (0.005s)
        params={
            "command_name": "motion",
            "asset_name": "robot",
            "stiffness": 600.0,
            "ang_stiffness": 120.0,
            "damping": 15.0,
            "axis_weights": (0.0, 0.0, 1.0),
            "gravity_comp": 0.5,
            "curriculum_factor": 1.0,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # motion_ended = DoneTerm(func=mdp.motion_ended, params={"command_name": "motion"})
    time_out = DoneTerm(func=mdp.my_time_out,
                        params={"command_name": "motion"},
                        time_out=True)
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        # params={"command_name": "motion", "threshold": 0.25},
        params={"command_name": "motion", "threshold": 0.3},

    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            # "threshold": 0.25,
            "threshold": 0.4,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )
    bad_anchor_pos_xy = DoneTerm(
        func=mdp.bad_anchor_pos_x_y_only,
        params={"command_name": "motion", "threshold": 0.7},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    spring_force_linear = CurrTerm(
        func=mdp.LinearForceScheduler,
        params={
            "command_name": "motion",
            "start_steps": 0,
            "ramp_steps": 240000,  # Ramp up over 10k iters (× 24 steps_per_env)
        },
    )
    spring_force_factor = CurrTerm(
        func=mdp.modify_term_cfg,
        params={
            "address": "events.assistive_spring_force.params.curriculum_factor",
            "modify_fn": mdp.linear_interpolate_fn,
            "modify_params": {
                "initial_value": 1.0,
                "final_value": 0.0,
                "difficulty_term_str": "spring_force_linear",
            },
        },
    )


##
# Environment configuration
##


@configclass
class Bones3ptEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=8192, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        # --- CLI flag overrides (via env vars set by train_bones.py) ---
        # PPO output mode
        ppo_output = os.environ.get("BONES_PPO_OUTPUT")
        if ppo_output in ("delta-pseudotarget", "delta-all"):
            self.actions.joint_pos = mdp.ReferenceJointPositionActionCfg(
                asset_name="robot", joint_names=[".*"], command_name="motion"
            )
            if ppo_output == "delta-pseudotarget":
                self.observations.policy.actions = ObsTerm(func=mdp.last_action_pseudotarget)
                self.observations.critic.actions = ObsTerm(func=mdp.last_action_pseudotarget)

        # Push perturbation (default: normal)
        push_mode = os.environ.get("BONES_PUSH", "normal")
        if push_mode != "none":
            vel_range = VELOCITY_RANGE_SOFT if push_mode == "soft" else VELOCITY_RANGE
            self.events.push_robot = EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(1.0, 3.0),
                params={"velocity_range": vel_range},
            )

        # Double-step penalty
        if os.environ.get("BONES_DOUBLE_STEP") == "1":
            self.rewards.double_step_penalty = RewTerm(
                func=mdp.double_step_penalty,
                weight=0.5,
                params={
                    "command_name": "motion",
                    "threshold": 2.0,
                    "body_names": ["left_ankle_roll_link", "right_ankle_roll_link"],
                },
            )

        # Crane mode: penalize foot on ground when reference says it should be in the air
        if os.environ.get("BONES_CRANE") == "1":
            self.rewards.foot_contact_state = RewTerm(
                func=mdp.foot_contact_state_penalty,
                weight=2.0,
                params={
                    "command_name": "motion",
                    "sensor_cfg": SceneEntityCfg(
                        "contact_forces",
                        body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                    ),
                    "height_threshold": 0.08,
                },
            )

        # Remove command observation
        if os.environ.get("BONES_NO_COMMAND_OBS") == "1":
            self.observations.policy.command = None

        # Assistive spring force curriculum
        if os.environ.get("BONES_CURRICULUM") == "1":
            self.events.assistive_spring_force = EventTerm(
                func=mdp.apply_spring_force,
                mode="interval",
                interval_range_s=(0.005, 0.005),
                params={
                    "command_name": "motion",
                    "asset_name": "robot",
                    "stiffness": 600.0,
                    "ang_stiffness": 300.0,
                    "damping": 15.0,
                    "axis_weights": (0.0, 0.0, 1.0),
                    "gravity_comp": 0.5,
                    "curriculum_factor": 1.0,
                },
            )
            self.curriculum = CurriculumCfg()

            assist_mode = os.environ.get("BONES_ASSIST_MODE", "both")
            if assist_mode not in {"both", "gravity_only", "spring_only", "none", "gravity_pelvis", "gravity_all", "both_pelvis", "both_all"}:
                raise ValueError(f"Unsupported BONES_ASSIST_MODE: {assist_mode}")

            # By default set mode to pelvis
            self.events.assistive_spring_force.params["gravity_comp_mode"] = "pelvis"

            if assist_mode == "gravity_pelvis":
                self.events.assistive_spring_force.params["stiffness"] = 0.0
                self.events.assistive_spring_force.params["damping"] = 0.0
                self.events.assistive_spring_force.params["ang_stiffness"] = 0.0
                self.events.assistive_spring_force.params["gravity_comp_mode"] = "pelvis"
            elif assist_mode == "gravity_all":
                self.events.assistive_spring_force.params["stiffness"] = 0.0
                self.events.assistive_spring_force.params["damping"] = 0.0
                self.events.assistive_spring_force.params["ang_stiffness"] = 0.0
                self.events.assistive_spring_force.params["gravity_comp_mode"] = "all"
            elif assist_mode == "both_pelvis":
                self.events.assistive_spring_force.params["gravity_comp_mode"] = "pelvis"
            elif assist_mode == "both_all":
                self.events.assistive_spring_force.params["gravity_comp_mode"] = "all"
            elif assist_mode == "spring_only":
                self.events.assistive_spring_force.params["gravity_comp"] = 0.0
            elif assist_mode == "none":
                self.events.assistive_spring_force.params["stiffness"] = 0.0
                self.events.assistive_spring_force.params["damping"] = 0.0
                self.events.assistive_spring_force.params["ang_stiffness"] = 0.0
                self.events.assistive_spring_force.params["gravity_comp"] = 0.0
