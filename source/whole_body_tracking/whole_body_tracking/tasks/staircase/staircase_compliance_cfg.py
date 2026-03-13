from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.staircase.mdp as mdp

# Recommended Implementation
from isaaclab.terrains.config import TerrainGeneratorCfg
import isaaclab.terrains.config as terrain_gen
from isaaclab.terrains import HfPyramidStairsTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPyramidStairsTerrainCfg

# from whole_body_tracking.tasks.chair_step.custom_terrains import LinearStairsTerrainCfg


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

VELOCITY_RANGE_Null = {
    "x": (-0.0, 0.0),
    "y": (-0.0, 0.0),
    "z": (-0.0, 0.0),
    "roll": (-0.0, 0.0),
    "pitch": (-0.0, 0.0),
    "yaw": (-0.0, 0.0),
}

# Staircase definition
STAIRCASE_POSITION = [0.0, 0.0, 0.0]


@configclass
class StaircaseSceneCfg(InteractiveSceneCfg):
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

    # Staircase object
    staircase = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Staircase",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase/multi_boxes_scaled_0.84_0.84_0.84.urdf",
            usd_dir=os.path.expanduser("~/tmp/IsaacLab/staircase_usd"),
            fix_base=True,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=STAIRCASE_POSITION),
    )

    # Depth camera mounted on the D435 link (head)
    # Only included when --enable_cameras is set (ENABLE_CAMERAS=1)
    depth_camera: TiledCameraCfg | None = (
        TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link/d435_link/depth_camera",
            update_period=0.1,  # 10Hz
            height=480,
            width=848,
            data_types=["rgb", "depth"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=1.93,  # D435i: ~87° HFOV
                horizontal_aperture=3.6,
                clipping_range=(0.1, 5.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, 0.0),  # Already positioned by d435_link in URDF
                rot=(0.5, -0.5, 0.5, -0.5),  # ROS convention: z-forward
                convention="ros",
            ),
        )
        if os.environ.get("ENABLE_CAMERAS", "0") == "1"
        else None
    )


##
# MDP settings
##


@configclass
class ComplianceObservationsCfg(BaseObservationsCfg):
    """Observations with CHIP compliance terms added."""

    @configclass
    class PolicyCfg(BaseObservationsCfg.PolicyCfg):
        """Policy observations extended with CHIP compliance."""

        # CHIP compliance observations
        compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})

    @configclass
    class PrivilegedCfg(BaseObservationsCfg.PrivilegedCfg):
        # CHIP compliance observations (critic gets both compliant and non-compliant targets)
        compliance = ObsTerm(func=mdp.compliance, params={"command_name": "motion"})
        vr_3point_pos_compliant = ObsTerm(func=mdp.vr_3point_local_compliant_target, params={"command_name": "motion"})
        vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: PrivilegedCfg = PrivilegedCfg()


@configclass
class ComplianceEventCfg(BaseEventCfg):
    """Events with CHIP compliance terms added."""

    # CHIP force push: apply randomized external forces to ankle/pelvis bodies
    force_push_robot = EventTerm(
        func=mdp.force_based_push,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "force_duration": [50, 100],
            "command_name": "motion",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )

    # CHIP compliance: randomize end-effector stiffness
    change_compliance = EventTerm(
        func=mdp.change_compliance,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "command_name": "motion",
            "compliance_lb": [0.0, 0.0, 0.0],
            "compliance_ub": [0.02, 0.02, 0.01],
            "compliance_duration": (100, 200),
            "start_steps": 0,
        },
    )


##
# Environment configuration
##


@configclass
class StaircaseComplianceCfg(ManagerBasedRLEnvCfg):
    """Configuration for the staircase environment with CHIP compliance."""

    # Scene settings
    scene: StaircaseSceneCfg = StaircaseSceneCfg(num_envs=4096, env_spacing=2.5)
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
