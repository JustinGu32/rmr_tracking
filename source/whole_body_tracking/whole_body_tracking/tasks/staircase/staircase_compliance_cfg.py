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

from whole_body_tracking.tasks.staircase.staircase_env_cfg import (
    StaircaseBaseCfg,
    StaircaseSceneCfg,
    ObservationsCfg as BaseObservationsCfg,
    EventCfg as BaseEventCfg,
    STAIRCASE_POSITION,
    VELOCITY_RANGE,
)


##
# CHIP Compliance overrides — only the parts that differ from the base staircase config
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
class StaircaseComplianceCfg(StaircaseBaseCfg):
    """Configuration for the staircase environment with CHIP compliance.

    Inherits the full ADR curriculum + assistive spring force event from
    StaircaseBaseCfg, and adds CHIP-specific observations and events.
    """

    # Override with CHIP observations and events
    observations: ComplianceObservationsCfg = ComplianceObservationsCfg()
    events: ComplianceEventCfg = ComplianceEventCfg()

    def __post_init__(self):
        super().__post_init__()

        # Compliance staircase uses fix_base=False (unlike base which uses True)
        self.scene.staircase = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Staircase",
            spawn=sim_utils.UrdfFileCfg(
                asset_path="/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase/multi_boxes_scaled_0.84_0.84_0.84.urdf",
                usd_dir=os.path.expanduser("~/tmp/IsaacLab/staircase_usd"),
                fix_base=False,
                collision_props=sim_utils.CollisionPropertiesCfg(),
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(articulation_enabled=False),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=STAIRCASE_POSITION),
        )
