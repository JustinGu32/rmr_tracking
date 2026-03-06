from __future__ import annotations

from dataclasses import MISSING
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.compliance.mdp as mdp
from whole_body_tracking.tasks.compliance.terrain_mesh_spawner_cfg import FLAT_PLANE_MARKER, MultiTerrainMeshSpawnerCfg

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


@configclass
class MultiTerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot and multiple mesh terrains from URDF files.

    Each environment will get a different terrain selected from a list of URDF files
    based on the environment ID.
    """

    # terrain as static mesh (one per environment, different URDF per environment)
    terrain: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Terrain",
        spawn=MultiTerrainMeshSpawnerCfg(
            urdf_paths=[],  # Will be set in __post_init__
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.4, 0.3)),
            physics_material=sim_utils.RigidBodyMaterialCfg(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                kinematic_enabled=True,  # Static terrain
            ),
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
    """Command specifications for the MDP (binded: each env bound to a motion via env_id % num_pairs)."""

    motion = mdp.BindedMultiMotionCommandCfg(
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


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        command = ObsTerm(func=mdp.command_lower_body, params={"command_name": "motion"})
        vr_3point_pos = ObsTerm(func=mdp.vr_3point_local_target, params={"command_name": "motion"})
        vr_3point_orn = ObsTerm(func=mdp.vr_3point_local_orn_target, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
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

    # Set collision groups per environment to prevent inter-environment collisions
    set_collision_groups = EventTerm(
        func=mdp.set_environment_collision_groups,
        mode="startup",
        params={
            "terrain_cfg": SceneEntityCfg("terrain"),
            "robot_cfg": SceneEntityCfg("robot"),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": VELOCITY_RANGE},
    )

    # force push robot
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

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    motion_ended = DoneTerm(func=mdp.motion_ended, params={"command_name": "motion"})
    anchor_pos = DoneTerm(
        func=mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25},
    )
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


def _resolve_path(path: Path, fallback_roots: list[Path]) -> Path:
    """Resolve a path, trying fallback roots if not found."""
    if path.is_absolute() and path.exists():
        return path.resolve()
    for root in fallback_roots:
        candidate = root / path
        if candidate.exists():
            return candidate.resolve()
    return path.resolve()


def _build_motion_terrain_pairs(motion_dir: str, terrain_folder_path: str) -> tuple[list[str], list[str]]:
    """Build paired lists of motion and terrain paths by matching base names (without extension).

    Pairs motion_dir/*.npz with terrain_folder/*.urdf where base names match.
    Motions without a matching terrain use FLAT_PLANE_MARKER (spawner creates default flat terrain).
    Returns (motion_files, urdf_paths) in matching order.
    """
    fallback_roots = [
        Path.cwd(),
        Path(__file__).parent.parent.parent.parent.parent.parent,  # bm_generalist root
    ]

    motion_path = _resolve_path(Path(motion_dir), fallback_roots)
    terrain_path = _resolve_path(Path(terrain_folder_path), fallback_roots)

    if not motion_path.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_path}")

    # Build map: base_name -> motion_file
    motion_by_base: dict[str, Path] = {}
    for f in motion_path.glob("*.npz"):
        motion_by_base[f.stem] = f

    if not motion_by_base:
        raise FileNotFoundError(f"No .npz motion files found in {motion_path}")

    # Build map: base_name -> terrain_file (empty if terrain folder missing or no matches)
    terrain_by_base: dict[str, Path] = {}
    if terrain_folder_path.strip() and terrain_path.exists():
        for f in terrain_path.glob("*.urdf"):
            terrain_by_base[f.stem] = f

    # Include ALL motions; use matching terrain or FLAT_PLANE_MARKER for default flat
    all_bases = sorted(motion_by_base.keys())
    motion_files = [str(motion_by_base[b].resolve()) for b in all_bases]
    urdf_paths = [
        str(terrain_by_base[b].resolve()) if b in terrain_by_base else FLAT_PLANE_MARKER
        for b in all_bases
    ]

    num_flat = sum(1 for u in urdf_paths if u == FLAT_PLANE_MARKER)
    if num_flat:
        print(f"[INFO]: {num_flat} motion(s) use default flat terrain (no matching .urdf)")

    return motion_files, urdf_paths


##
# Environment configuration
##


@configclass
class Tracking3ptBindedMultiTerrainEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for 3pt tracking with binded motion-terrain pairs.

    Each environment uses:
    - Motion from motion_dir (e.g. run1.npz)
    - Terrain from terrain_folder_path (e.g. run1.urdf)
    Pairs are matched by base name (without extension). Env i gets pair i (env_id % num_pairs).
    """

    motion_dir: str = ""
    terrain_folder_path: str = ""

    scene: InteractiveSceneCfg = MISSING
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def setup_motion_terrain_pairs(self):
        """Build motion-terrain pairs from motion_dir and terrain_folder_path.

        Call this after setting motion_dir and terrain_folder_path (e.g. from CLI).
        Sets commands.motion.motion_files, scene.terrain.spawn.urdf_paths, and scene.num_envs.
        """
        motion_files, urdf_paths = _build_motion_terrain_pairs(
            self.motion_dir,
            self.terrain_folder_path,
        )
        self.commands.motion.motion_files = motion_files
        self.scene.terrain.spawn.urdf_paths = urdf_paths
        self.scene.terrain.spawn.num_envs = len(motion_files)
        self.scene.num_envs = len(motion_files)
        print(f"[INFO]: Paired {len(motion_files)} motion-terrain files (name-matched)")
        print(f"[INFO]: Motions: {[Path(p).name for p in motion_files]}")
        print(f"[INFO]: Terrains: {[Path(p).name for p in urdf_paths]}")

    def __post_init__(self):
        """Post initialization."""
        has_paths = bool(self.motion_dir.strip()) and bool(self.terrain_folder_path.strip())
        if not has_paths:
            # Paths must be provided via CLI; use placeholder until setup_motion_terrain_pairs() is called
            self.scene = MultiTerrainSceneCfg(num_envs=1, env_spacing=10.0)
            self.scene.terrain.spawn.urdf_paths = []
            self.scene.terrain.spawn.num_envs = 1
            self.commands.motion.motion_files = []
            print(
                "[INFO]: motion_dir and terrain_folder_path required (e.g. --motion_dir, --terrain_folder). "
                "Call setup_motion_terrain_pairs() after setting paths."
            )
        else:
            fallback_roots = [
                Path.cwd(),
                Path(__file__).parent.parent.parent.parent.parent.parent,  # bm_generalist root
            ]
            motion_path = _resolve_path(Path(self.motion_dir), fallback_roots)
            terrain_path = _resolve_path(Path(self.terrain_folder_path), fallback_roots)

            if motion_path.exists() and terrain_path.exists():
                motion_files, urdf_paths = _build_motion_terrain_pairs(
                    self.motion_dir,
                    self.terrain_folder_path,
                )
                num_envs = len(motion_files)
                self.scene = MultiTerrainSceneCfg(num_envs=num_envs, env_spacing=10.0)
                self.scene.terrain.spawn.urdf_paths = urdf_paths
                self.scene.terrain.spawn.num_envs = num_envs
                self.commands.motion.motion_files = motion_files
                print(f"[INFO]: Paired {num_envs} motion-terrain files (name-matched)")
                print(f"[INFO]: Motions: {[Path(p).name for p in motion_files]}")
                print(f"[INFO]: Terrains: {[Path(p).name for p in urdf_paths]}")
            else:
                self.scene = MultiTerrainSceneCfg(num_envs=1, env_spacing=10.0)
                self.scene.terrain.spawn.urdf_paths = []
                self.scene.terrain.spawn.num_envs = 1
                self.commands.motion.motion_files = []
                print(
                    f"[INFO]: Motion/terrain paths not found (motion_dir={self.motion_dir}, "
                    f"terrain_folder={self.terrain_folder_path}). "
                    "Call setup_motion_terrain_pairs() after setting paths (e.g. from CLI)."
                )

        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physx.gpu_collision_stack_size = 2**28
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
