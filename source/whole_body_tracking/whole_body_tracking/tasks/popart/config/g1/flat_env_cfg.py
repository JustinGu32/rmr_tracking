from isaaclab.utils import configclass

from whole_body_tracking.robots.g1 import G1_ACTION_SCALE, G1_CYLINDER_CFG
from whole_body_tracking.tasks.popart.popart_env_cfg import PopartEnvCfg
from whole_body_tracking.tasks.popart.mdp.commands import MultiClipMotionCommandPopArtCfg


@configclass
class G1FlatEnvCfgPopArt(PopartEnvCfg):
    """G1 flat tracking with multi-clip Zarr motion loading + per-env category index.

    Stage A: identical to G1FlatMultiClipEnvCfg in the tracking task, plus the
    `category` observation group exposed by `PopartEnvCfg`. The PopArt-aware
    ActorCritic / PPO is wired in Stage B.
    """

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

        # Replace the single-file MotionCommand with the PopArt multi-clip variant.
        # All shared fields are forwarded from the parent CommandsCfg's instance.
        parent_cfg = self.commands.motion
        # diffusion_collect group is a leftover from collect-mode; not used in RL training
        self.observations.diffusion_collect = None

        self.commands.motion = MultiClipMotionCommandPopArtCfg(
            asset_name=parent_cfg.asset_name,
            resampling_time_range=parent_cfg.resampling_time_range,
            debug_vis=parent_cfg.debug_vis,
            zarr_path="",  # set at runtime by train.py via --zarr_path
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
            # Single source of truth for category structure: the `categories` list.
            # Drives num_categories=len(list), the priority-substring categorizer,
            # and (when include_motion_types isn't set) the zarr clip filter.
            # List order = matching priority — put more-specific names first.
            # Override at the CLI with --categories=walk,stand_up etc.
            categories=["walk", "jog"],
            unmatched="raise",
        )

        # Jump-exploration sampling/RSI knobs, forwarded from the WBT_* env vars
        # set by train_bones.py flags (--error_blend_beta / --flight_rsi_ratio).
        # Set here (after the command cfg is rebuilt) so they aren't lost when the
        # parent's command cfg is replaced above. Default 0.0 = inert.
        import os as _os
        self.commands.motion.error_blend_beta = float(_os.environ.get("WBT_ERROR_BLEND_BETA", "0.0"))
        self.commands.motion.flight_rsi_ratio = float(_os.environ.get("WBT_FLIGHT_RSI_RATIO", "0.0"))


@configclass
class G1FlatEnvCfgPopArt_PLAY(G1FlatEnvCfgPopArt):
    """Play-mode config: longer episodes, no random pushes / curricula / hard terminations."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 60.0
        # The base PopartEnvCfg may not have these attributes — guard with hasattr.
        if hasattr(self.curriculum, "adr"):
            self.curriculum.adr = None
        if hasattr(self.curriculum, "spring_force_adr"):
            self.curriculum.spring_force_adr = None
        if hasattr(self, "spring_force_cfg"):
            self.spring_force_cfg = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
        if hasattr(self.events, "force_push_robot"):
            self.events.force_push_robot = None
        if hasattr(self.terminations, "bad_anchor_pos_xy"):
            self.terminations.bad_anchor_pos_xy = None
        if hasattr(self.terminations, "ee_body_pos"):
            self.terminations.ee_body_pos = None
        if hasattr(self.terminations, "anchor_pos"):
            self.terminations.anchor_pos = None
            
