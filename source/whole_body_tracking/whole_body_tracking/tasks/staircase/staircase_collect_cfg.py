from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.staircase.mdp as mdp

from .staircase_env_cfg import ObservationsCfg, StaircaseEnvCfg


@configclass
class StaircaseCollectObservationsCfg(ObservationsCfg):
    @configclass
    class DiffusionCollect(ObsGroup):
        """Observation group used by dataset collection."""

        body_pos = ObsTerm(func=mdp.robot_body_pos_w, noise=Unoise(n_min=-0.05, n_max=0.05))
        body_ori = ObsTerm(func=mdp.robot_body_ori_w_quat, noise=Unoise(n_min=-0.02, n_max=0.02))
        body_lin_vel = ObsTerm(func=mdp.robot_body_lin_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))
        body_ang_vel = ObsTerm(func=mdp.robot_body_ang_vel_w, noise=Unoise(n_min=-0.2, n_max=0.2))
        dof_pos = ObsTerm(func=mdp.joint_pos_rel)
        dof_vel = ObsTerm(func=mdp.joint_vel_rel)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    diffusion_collect: DiffusionCollect = DiffusionCollect()


@configclass
class StaircaseCollectCfg(StaircaseEnvCfg):
    """Staircase environment configured for dataset collection."""

    observations: StaircaseCollectObservationsCfg = StaircaseCollectObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.episode_length_s = 100.0
