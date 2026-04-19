from __future__ import annotations

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.staircase.mdp as mdp
from whole_body_tracking.tasks.staircase.staircase_env_cfg import (
    ActionsCfg,
    CommandsCfg,
    CurriculumCfg,
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    StaircaseSceneCfg,
    TerminationsCfg,
)


@configclass
class StaircaseCollectObservationsCfg(ObservationsCfg):
    """Observation config for staircase data collection."""

    @configclass
    class DiffusionCollect(ObsGroup):
        body_pos = ObsTerm(func=mdp.robot_body_pos_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        body_ori = ObsTerm(func=mdp.robot_body_ori_w_quat, noise=Unoise(n_min=-0.02, n_max=0.02))
        body_lin_vel = ObsTerm(func=mdp.robot_body_lin_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))
        body_ang_vel = ObsTerm(func=mdp.robot_body_ang_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))
        dof_pos = ObsTerm(func=mdp.joint_pos_rel)
        dof_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    diffusion_collect: DiffusionCollect = DiffusionCollect()


@configclass
class StaircaseCollectCfg(ManagerBasedRLEnvCfg):
    """Collection-friendly staircase environment configuration."""

    scene: StaircaseSceneCfg = StaircaseSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: StaircaseCollectObservationsCfg = StaircaseCollectObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg | None = None

    def __post_init__(self):
        self.decimation = 6
        self.episode_length_s = 100.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
