from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# G1 foot bodies, used for flight detection / foot-clearance jump terms.
ANKLE_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        # import ipdb; ipdb.set_trace() 
             
        self.min_sample_idx = cfg.min_sample_idx
        self.max_sample_idx = cfg.max_sample_idx
        self.steps_collect = cfg.steps_collect

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # VR 3-point related (feet + pelvis tracking points)
        self.vr_3point_body_indices = [self.robot.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_indices_motion = [self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_offsets = torch.tensor(self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device).view(1, -1, 3).repeat(self.num_envs, 1, 1)

        self.down_dir = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).view(1, -1, 3).repeat(self.num_envs, 1, 1)

        # Force push related (CHIP)
        self.force_update_frequency = self.cfg.force_update_frequency
        self.max_force = self.cfg.max_force
        self.num_bodies = len(self.cfg.body_names)
        self.body_force_dir_buf = torch.randn(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.body_force_dir_buf /= torch.norm(self.body_force_dir_buf, dim=-1, keepdim=True)
        self.body_force_magnitude_buf = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)

        self.force_push_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.force_duration_per_env = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.force_config_init = False
        self.force_push_ids = self.robot.find_bodies(self.cfg.force_push_body, preserve_order=True)[0]
        self.non_force_push_ids_rel = []
        self.force_push_ids_rel = []
        for i, idx in enumerate(self.body_indexes.tolist()):
            if idx not in self.force_push_ids:
                self.non_force_push_ids_rel.append(i)
            else:
                self.force_push_ids_rel.append(i)

        self.force_push_body_offsets = torch.tensor(self.cfg.force_push_body_offset, dtype=torch.float32, device=self.device).view(1, -1, 3).repeat(self.num_envs, 1, 1)
        self.last_force_applied = torch.zeros(self.num_envs, len(self.force_push_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)

        # Compliance related (CHIP)
        self.compliance_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.compliance_duration_per_env = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.eef_stiffness_buf = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.compliance_config_init = False

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_applied"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    # Takara 
    def reload_motion(self):
        # import ipdb; ipdb.set_trace() 
        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)

    # TODO: may not need?
    @property
    def joint_pos_ref(self) -> torch.Tensor:
        return self.motion.joint_pos #[self.time_steps]
    
    # TODO: may not need?
    @property
    def joint_vel_ref(self) -> torch.Tensor:
        return self.motion.joint_vel #[self.time_steps]

    
    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    # TODO: may not need?
    #Takara
    @property
    def ref_pos_w(self) -> torch.Tensor:
        # import ipdb; ipdb.set_trace()
        return self.motion._body_pos_w[self.time_steps] + self._env.scene.env_origins[:,None,:]
    # TODO: may not need?

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self.motion._body_quat_w[self.time_steps]
    # TODO: may not need?

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self.motion._body_lin_vel_w[self.time_steps] 
    # TODO: may not need?
    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self.motion._body_ang_vel_w[self.time_steps]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    # VR 3-point properties (CHIP compliance tracking points)
    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps][:, self.vr_3point_body_indices_motion]

    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps][:, self.vr_3point_body_indices_motion] \
            + quat_apply(self.vr_3point_body_quat_w, self.vr_3point_body_offsets) \
            + self._env.scene.env_origins[:, None, :]

    @property
    def robot_vr_3point_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.vr_3point_body_indices]

    @property
    def robot_vr_3point_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.vr_3point_body_indices] \
            + quat_apply(self.robot_vr_3point_quat_w, self.vr_3point_body_offsets)

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        self.metrics["force_applied"] = torch.norm(self.last_force_applied, dim=-1).mean(dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        self.time_steps[env_ids] = torch.clamp(
            self.time_steps[env_ids],
            min=self.min_sample_idx,
            max=min(self.max_sample_idx, self.motion.time_step_total - 1),
        )
        
        # eps_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        # self.time_steps[env_ids[eps_mask]] = 0

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _uniform_sampling(self, env_ids: Sequence[int]):
        # Reference State Initialization: each env starts at a uniformly random frame of the clip.
        phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        time_samples = (phase * (self.motion.time_step_total - 1)).long()
        # Only restrict to a sub-window [min_sample_idx, max_sample_idx - steps_collect] when a
        # real window is configured. NOTE: guarding on `is not None` is wrong because the default
        # max_sample_idx=0/steps_collect=0 collapses the range to a single frame (always frame 0).

        if self.max_sample_idx:
            sampling_range = (self.max_sample_idx - self.steps_collect) - self.min_sample_idx
            time_samples = (phase * sampling_range + self.min_sample_idx).long()
            time_samples = torch.clip(time_samples, min=self.min_sample_idx, max=self.max_sample_idx - self.steps_collect)

        self.time_steps[env_ids] = time_samples.long()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        # TODO: if training, use adaptive sampling (make a command-line flag)
        self._adaptive_sampling(env_ids)

        # self._uniform_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone() 
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.2
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    min_sample_idx: int = 0
    max_sample_idx: int = 10**9
    steps_collect: int = 1

    # CHIP force push config
    force_update_frequency: int = 100
    max_force: float = 20.0

    force_push_body: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link", "pelvis"]
    force_push_body_offset: list[list[float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    # CHIP VR 3-point tracking config
    vr_3point_body: list[str] = ["left_ankle_roll_link", "right_ankle_roll_link", "pelvis"]
    vr_3point_body_offset: list[list[float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


class ZarrMotionLoader:
    """Load motions from a Zarr store containing multiple clips.

    All data is loaded into CPU pinned memory. Per-env indexing is done at
    training time by MultiClipMotionCommand.
    """

    def __init__(self, zarr_path: str, body_indexes: Sequence[int], device: str = "cpu",
                 exclude_props: list[str] | None = None, max_clips: int | None = None,
                 include_keywords: list[str] | None = None,
                 include_clip_names: list[str] | None = None):
        import zarr as _zarr

        assert os.path.isdir(zarr_path), f"Invalid zarr path: {zarr_path}"
        store = _zarr.open(zarr_path, mode="r")

        self.fps = int(store["fps"][0])
        all_clip_start = store["clip_start_idx"][:]
        all_clip_end = store["clip_end_idx"][:]
        total_clips_raw = len(all_clip_start)

        # Filter clips by content_props_desc if requested
        if exclude_props and "content_props_desc" in store:
            desc = store["content_props_desc"][:]
            valid_mask = []
            for i, d in enumerate(desc):
                d_str = str(d).strip().lower()
                excluded = any(ep.lower() in d_str for ep in exclude_props)
                valid_mask.append(not excluded)
            valid_indices = [i for i, v in enumerate(valid_mask) if v]
            excluded_count = total_clips_raw - len(valid_indices)
            print(f"[ZarrMotionLoader] Excluded {excluded_count}/{total_clips_raw} clips "
                  f"matching props: {exclude_props}")
        else:
            valid_indices = list(range(total_clips_raw))

        # Filter clips by clip_name keywords if requested (case-insensitive substring,
        # OR semantics: keep a clip if any keyword matches). Applied after exclude_props.
        # Always read clip_names if present so we can retain them post-filter for the
        # categorizer (used by MultiClipMotionCommandCategorized).
        all_names = store["clip_names"][:] if "clip_names" in store else None
        if include_keywords and all_names is not None:
            kws = [kw.lower() for kw in include_keywords]
            before = len(valid_indices)
            valid_indices = [
                i for i in valid_indices
                if any(kw in str(all_names[i]).lower() for kw in kws)
            ]
            print(f"[ZarrMotionLoader] Kept {len(valid_indices)}/{before} clips "
                  f"matching keywords: {include_keywords}")

        # Filter clips by exact clip name list if requested (case-sensitive, set-membership).
        # Applied AFTER `include_keywords` so it's an intersection — a clip must pass both
        # the keyword filter and the exact-name filter. Used for specialist training and
        # DAgger on a precomputed failed-clip subset.
        if include_clip_names and all_names is not None:
            name_set = set(include_clip_names)
            before = len(valid_indices)
            before_sample = [str(all_names[i]) for i in valid_indices[:10]]
            valid_indices = [
                i for i in valid_indices
                if str(all_names[i]) in name_set
            ]
            print(f"[ZarrMotionLoader] Kept {len(valid_indices)}/{before} clips "
                  f"matching exact names from list of {len(name_set)}")
            if len(valid_indices) == 0:
                raise RuntimeError(
                    f"include_clip_names filter produced 0 clips. "
                    f"Filter sample: {list(name_set)[:5]}. "
                    f"Zarr sample (post-keyword-filter): {before_sample}"
                )

        # Limit number of clips if requested
        if max_clips is not None and max_clips < len(valid_indices):
            valid_indices = valid_indices[:max_clips]
            print(f"[ZarrMotionLoader] Limited to {max_clips} clips")

        self.clip_start_idx = torch.tensor([all_clip_start[i] for i in valid_indices], dtype=torch.long)
        self.clip_end_idx = torch.tensor([all_clip_end[i] for i in valid_indices], dtype=torch.long)
        self.num_clips = len(self.clip_start_idx)
        self.clip_lengths = self.clip_end_idx - self.clip_start_idx
        # Post-filter clip names, aligned by index with clip_start_idx / clip_end_idx.
        if all_names is not None:
            self.clip_names = [str(all_names[i]) for i in valid_indices]
        else:
            self.clip_names = [f"clip_{i}" for i in range(len(valid_indices))]

        # Only load frames that are actually referenced by the selected clips
        if max_clips is not None and max_clips < total_clips_raw:
            frame_end = int(self.clip_end_idx.max().item())
            print(f"[ZarrMotionLoader] Loading only first {frame_end} frames (of {store['joint_pos'].shape[0]})")
        else:
            frame_end = store['joint_pos'].shape[0]

        # Load data to the specified device (GPU if available)
        self.joint_pos = torch.tensor(store["joint_pos"][:frame_end], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(store["joint_vel"][:frame_end], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(store["body_pos_w"][:frame_end], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(store["body_quat_w"][:frame_end], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(store["body_lin_vel_w"][:frame_end], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(store["body_ang_vel_w"][:frame_end], dtype=torch.float32, device=device)
        # Body indexes from robot.find_bodies() — Zarr is already in Isaac order
        self._body_indexes = body_indexes.to(device) if isinstance(body_indexes, torch.Tensor) else torch.tensor(body_indexes, dtype=torch.long, device=device)
        self.time_step_total = self.joint_pos.shape[0]

        print(f"[ZarrMotionLoader] Loaded {self.num_clips} clips, "
              f"{self.time_step_total} total frames @ {self.fps} fps, "
              f"{self._body_pos_w.shape[1]} bodies, device={device}")

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MultiClipMotionCommand(MotionCommand):
    """Motion command that samples from multiple motion clips stored in a Zarr
    archive. Each environment tracks an independently sampled clip.

    Inherits from MotionCommand and overrides only the parts that differ:
    motion loading, clip selection on reset, and the per-env time-step logic.
    """

    cfg: "MultiClipMotionCommandCfg"

    def __init__(self, cfg: "MultiClipMotionCommandCfg", env: "ManagerBasedRLEnv"):
        # Skip MotionCommand.__init__ — we'll set up everything ourselves
        # because the parent tries to load a single-file MotionLoader.
        CommandTerm.__init__(self, cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        print(f"[DEBUG] Isaac body_names ({len(self.robot.body_names)}): {list(self.robot.body_names)}")
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
            dtype=torch.long, device=self.device,
        )
        print(f"[DEBUG] Isaac joint_names ({len(self.robot.joint_names)}): {list(self.robot.joint_names)}")

        self.min_sample_idx = cfg.min_sample_idx
        self.max_sample_idx = cfg.max_sample_idx
        self.steps_collect = cfg.steps_collect

        # --- Multi-clip specific: load from Zarr directly to GPU ---
        exclude_props = [
            "object manipulation", "wall", "chair", "obstacle",
            "edge", "safety pad", "railing", "box",
        ] if self.cfg.exclude_objects else None
        self.motion = ZarrMotionLoader(self.cfg.zarr_path, self.body_indexes, device=self.device,
                                       exclude_props=exclude_props,
                                       max_clips=self.cfg.max_clips,
                                       include_keywords=self.cfg.include_motion_types,
                                       include_clip_names=self.cfg.include_clip_names)

        # Per-env state: which clip each env is tracking and the absolute time step
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Keep motion data on CPU pinned memory (too large for GPU at full scale).
        # Only clip indices (tiny) go to GPU. Per-env frame slices are
        # transferred to GPU on-the-fly in the property accessors.
        self.motion.clip_start_idx = self.motion.clip_start_idx.to(self.device)
        self.motion.clip_end_idx = self.motion.clip_end_idx.to(self.device)
        self.motion.clip_lengths = self.motion.clip_lengths.to(self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # Adaptive sampling: flat global-timeline bins (same approach as single-clip)
        # Treat entire Zarr store as one concatenated sequence
        self.total_frames = int(self.motion.clip_end_idx.max().item())
        self.bin_count = max(self.total_frames // 50, 100)  # ~50 frames per bin
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        # VR 3-point related
        self.vr_3point_body_indices = [self.robot.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_indices_motion = [self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_offsets = torch.tensor(
            self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device
        ).view(1, -1, 3).repeat(self.num_envs, 1, 1)

        self.down_dir = torch.tensor(
            [0.0, 0.0, -1.0], dtype=torch.float32, device=self.device
        ).view(1, -1, 3).repeat(self.num_envs, 1, 1)

        # Force push related
        self.force_update_frequency = self.cfg.force_update_frequency
        self.max_force = self.cfg.max_force
        self.num_bodies = len(self.cfg.body_names)
        self.body_force_dir_buf = torch.randn(
            self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.body_force_dir_buf /= torch.norm(self.body_force_dir_buf, dim=-1, keepdim=True)
        self.body_force_magnitude_buf = torch.rand(
            self.num_envs, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.force_push_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.force_duration_per_env = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.force_config_init = False
        self.force_push_ids = self.robot.find_bodies(self.cfg.force_push_body, preserve_order=True)[0]
        self.non_force_push_ids_rel = []
        self.force_push_ids_rel = []
        for i, idx in enumerate(self.body_indexes.tolist()):
            if idx not in self.force_push_ids:
                self.non_force_push_ids_rel.append(i)
            else:
                self.force_push_ids_rel.append(i)

        self.force_push_body_offsets = torch.tensor(
            self.cfg.force_push_body_offset, dtype=torch.float32, device=self.device
        ).view(1, -1, 3).repeat(self.num_envs, 1, 1)
        self.last_force_applied = torch.zeros(
            self.num_envs, len(self.force_push_ids), 3, dtype=torch.float, device=self.device, requires_grad=False
        )

        # Compliance related
        self.compliance_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.compliance_duration_per_env = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.eef_stiffness_buf = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.compliance_config_init = False

        # Metrics
        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["force_applied"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Sampling mode: "adaptive" (default) or "uniform", set via BONES_SAMPLING env var in train script
        self._use_adaptive = os.environ.get("BONES_SAMPLING", "adaptive") == "adaptive"

        # Initial clip assignment (also initializes the per-step frame cache)
        self._assign_random_clips(torch.arange(self.num_envs, device=self.device))

    # ── Per-step frame cache ────────────────────────────────────────────

    def _cache_current_frames(self):
        """Pre-fetch all motion data for current time_steps once per step.

        This eliminates ~20 redundant scatter-gather GPU index operations
        that previously occurred across property accessors, rewards,
        observations, terminations, and _update_command().
        """
        ts = self.time_steps
        # Filtered body arrays (body_indexes subset)
        self._cached_joint_pos = self.motion.joint_pos[ts]
        self._cached_joint_vel = self.motion.joint_vel[ts]
        self._cached_body_pos_w = self.motion.body_pos_w[ts]
        self._cached_body_quat_w = self.motion.body_quat_w[ts]
        self._cached_body_lin_vel_w = self.motion.body_lin_vel_w[ts]
        self._cached_body_ang_vel_w = self.motion.body_ang_vel_w[ts]
        # Unfiltered body arrays (for ref_* properties)
        self._cached_raw_body_pos_w = self.motion._body_pos_w[ts]
        self._cached_raw_body_quat_w = self.motion._body_quat_w[ts]
        self._cached_raw_body_lin_vel_w = self.motion._body_lin_vel_w[ts]
        self._cached_raw_body_ang_vel_w = self.motion._body_ang_vel_w[ts]

        # Future reference frames (for anticipatory observations)
        if self.cfg.future_steps:
            self._cached_future_joint_pos = []
            self._cached_future_body_pos_w = []
            self._cached_future_body_quat_w = []
            for offset in self.cfg.future_steps:
                future_ts = torch.clamp(ts + offset, max=self.clip_end - 1)
                self._cached_future_joint_pos.append(self.motion.joint_pos[future_ts])
                self._cached_future_body_pos_w.append(self.motion.body_pos_w[future_ts])
                self._cached_future_body_quat_w.append(self.motion.body_quat_w[future_ts])

    # ── Property overrides: read from per-step cache ──────────────────

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._cached_joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._cached_joint_vel

    @property
    def command(self) -> torch.Tensor:
        """Current ref joint_pos + joint_vel, plus future ref joint_pos if configured."""
        parts = [self._cached_joint_pos, self._cached_joint_vel]
        if self.cfg.future_steps:
            for i in range(len(self.cfg.future_steps)):
                parts.append(self._cached_future_joint_pos[i])
        return torch.cat(parts, dim=1)

    @property
    def clip_phase(self) -> torch.Tensor:
        """Normalized progress through current clip: 0.0 = start, 1.0 = end. Shape (num_envs, 1)."""
        progress = (self.time_steps - self.clip_start).float() / (self.clip_end - self.clip_start).float().clamp(min=1)
        return progress.unsqueeze(-1)

    @property
    def time_to_live(self) -> torch.Tensor:
        """Time remaining in current clip in seconds. Shape (num_envs, 1)."""
        remaining_steps = (self.clip_end - self.time_steps).clamp(min=0).float()
        return (remaining_steps / self.motion.fps).unsqueeze(-1)

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._cached_body_pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._cached_body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._cached_body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._cached_body_ang_vel_w

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self._cached_body_pos_w[:, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self._cached_body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self._cached_body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self._cached_body_ang_vel_w[:, self.motion_anchor_body_index]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self._cached_raw_body_pos_w + self._env.scene.env_origins[:, None, :]

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self._cached_raw_body_quat_w

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self._cached_raw_body_lin_vel_w

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self._cached_raw_body_ang_vel_w

    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        return self._cached_body_quat_w[:, self.vr_3point_body_indices_motion]

    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        return self._cached_body_pos_w[:, self.vr_3point_body_indices_motion] \
            + quat_apply(self.vr_3point_body_quat_w, self.vr_3point_body_offsets) \
            + self._env.scene.env_origins[:, None, :]

    def get_future_ref_frames(self):
        """Return cached future reference frames.

        Returns:
            list of (joint_pos, body_pos_w, body_quat_w) tuples, one per future_steps offset.
            body_pos_w includes env_origins offset.
            Returns empty list if future_steps is not configured.
        """
        if not self.cfg.future_steps:
            return []
        result = []
        for i in range(len(self.cfg.future_steps)):
            result.append((
                self._cached_future_joint_pos[i],
                self._cached_future_body_pos_w[i] + self._env.scene.env_origins[:, None, :],
                self._cached_future_body_quat_w[i],
            ))
        return result

    def _assign_random_clips(self, env_ids: torch.Tensor):
        """Assign clips and start frames to the given envs by uniform sampling
        over the global timeline. Clips are effectively weighted by length, so
        each frame across the full dataset is equally likely to be picked."""
        n = len(env_ids)
        global_idx = torch.randint(0, self.total_frames, (n,), device=self.device)
        clip_ids = torch.searchsorted(self.motion.clip_end_idx, global_idx, right=True)
        clip_ids = torch.clamp(clip_ids, max=self.motion.num_clips - 1)
        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = self.motion.clip_start_idx[clip_ids]
        self.clip_end[env_ids] = self.motion.clip_end_idx[clip_ids]
        self.time_steps[env_ids] = global_idx

        # Refresh cache for newly assigned envs
        self._cache_current_frames()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device)

        # Track failures by global timestep bin
        episode_failed = self._env.termination_manager.terminated[env_ids_t]
        if torch.any(episode_failed):
            fail_bins = torch.clamp(
                (self.time_steps[env_ids_t][episode_failed] * self.bin_count) // max(self.total_frames, 1),
                0, self.bin_count - 1,
            )
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count).float()

        # Sampling over global timeline (adaptive vs uniform random clip assignment)
        if self._use_adaptive:
            self._adaptive_sampling(env_ids_t)
        else:
            self._assign_random_clips(env_ids_t)

        # Reset robot state (same as parent)
        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        """Sample start positions from failure-weighted global timeline bins."""
        n = len(env_ids)

        # Build sampling distribution from failure histogram + uniform baseline
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            sampling_probabilities, self.kernel.view(1, 1, -1)
        ).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        # Sample global bins → global frame indices
        sampled_bins = torch.multinomial(sampling_probabilities, n, replacement=True)
        global_frames = (
            (sampled_bins.float() + torch.rand(n, device=self.device))
            / self.bin_count
            * self.total_frames
        ).long().clamp(0, self.total_frames - 1)

        # Map global frame indices → clip IDs via searchsorted
        # clip_end_idx is sorted and contiguous, so searchsorted gives the clip index
        clip_ids = torch.searchsorted(self.motion.clip_end_idx, global_frames, right=True)
        clip_ids = clip_ids.clamp(0, self.motion.num_clips - 1)

        # Set clip state
        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = self.motion.clip_start_idx[clip_ids]
        self.clip_end[env_ids] = self.motion.clip_end_idx[clip_ids]

        # Clamp global frames within their clip boundaries
        self.time_steps[env_ids] = global_frames.clamp(
            self.clip_start[env_ids], self.clip_end[env_ids] - 1
        )

        # Refresh cache
        self._cache_current_frames()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _update_command(self):
        self.time_steps += 1
        # Check which envs have exceeded their clip boundary
        env_ids = torch.where(self.time_steps >= self.clip_end)[0]
        self._resample_command(env_ids)

        # Refresh the per-step frame cache (one batch of GPU gathers for the whole step)
        self._cache_current_frames()

        n_bodies = len(self.cfg.body_names)
        # Use .expand() (zero-copy views) instead of .repeat() where tensors are only read
        anchor_pos_w_exp = self.anchor_pos_w[:, None, :].expand(-1, n_bodies, -1)
        anchor_quat_w_exp = self.anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)
        robot_anchor_quat_w_exp = self.robot_anchor_quat_w[:, None, :].expand(-1, n_bodies, -1)

        # delta_pos_w is mutated in-place, so it needs a real copy
        delta_pos_w = self.robot_anchor_pos_w[:, None, :].expand(-1, n_bodies, -1).clone()
        delta_pos_w[..., 2] = anchor_pos_w_exp[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_exp, quat_inv(anchor_quat_w_exp)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_exp)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

@configclass
class MultiClipMotionCommandCfg(MotionCommandCfg):
    """Configuration for multi-clip motion command using a Zarr store."""

    class_type: type = MultiClipMotionCommand

    zarr_path: str = MISSING
    """Path to the Zarr motion store."""

    exclude_objects: bool = True
    """Whether to exclude motions whose content_props_desc contains any scene
    prop (object manipulation / wall / chair / obstacle / edge / safety pad /
    railing / box). Default: True."""

    max_clips: int | None = None
    """Maximum number of clips to load. None = load all. Useful for play/eval on smaller GPUs."""

    include_motion_types: list[str] | None = None
    """Optional list of keywords; keep only clips whose `clip_names` contains any
    keyword (case-insensitive substring, OR semantics). Applied after
    `exclude_objects`. None = no name-based filtering. Examples: ["walk", "jog"],
    ["jump"]. Multi-label by construction — a clip named "Turn_Start_Walk" matches
    both "turn" and "walk"."""

    include_clip_names: list[str] | None = None
    """Optional list of EXACT clip names to keep (case-sensitive, set-membership).
    Applied AFTER `include_motion_types` (intersection — a clip must pass both filters).
    None = no exact-name filtering. Used by specialist training and DAgger to restrict
    the env to a precomputed failed-clip subset. Plumbed via the CLI flag
    `--include_clip_names_file <path-to-json-list>` in train_bones.py / dagger.py."""

    future_steps: list[int] = []
    """Future timestep offsets to include in observations (e.g., [5, 10, 15]).
    Empty list = no future frames cached. Each offset is clamped to clip boundaries."""

    # Override motion_file — not used in multi-clip mode
    motion_file: str = ""


# ─── Categorized multi-clip command ──────────────────────────────────────────
# Adds a per-env category index, derived from clip name via a named categorizer.
# Used internally for category-aware adaptive sampling (cat_blend_clip_uniform,
# cat_adaptive_clip_uniform, etc). Does NOT expose the category as an
# observation here — that's only needed by the PopArt-aware ActorCritic.

from whole_body_tracking.tasks.generalist.mdp.categorizers import (  # noqa: E402
    CATEGORIZERS,
    make_priority_categorizer,
    names_for,
)


def _mix_uniform(score: torch.Tensor, p: float) -> torch.Tensor:
    """Blend an adaptive failure score with the uniform distribution.

    Returns sampling weights ``(1 - p) * (score / score.sum()) + p * (1 / N)``
    over the support of size ``N = score.numel()``. ``p`` is the probability mass
    placed on UNIFORM sampling (the `*_uniform_prob` cfg fields):
        p = 0  -> pure failure-adaptive
        p = 1  -> fully uniform
    Degenerates to uniform when the score is all-zero (early training, before any
    failures have accumulated). `score` must be 1-D and non-negative.

    Replaces the old additive-floor form ``score + ratio/N`` (renormalized),
    where ``ratio`` was an unbounded, K-dependent weight; here ``p`` is the
    interpretable, bounded mixture weight, with ``p = ratio/(N+ratio)``.
    """
    n = score.numel()
    s = score.sum()
    if float(s) > 0.0:
        adapt = score / s
    else:
        adapt = torch.full_like(score, 1.0 / n)
    return (1.0 - p) * adapt + p / n


def _additive_floor(score: torch.Tensor, ratio: float) -> torch.Tensor:
    """OLD additive-floor sampler (restored for backwards compat with runs
    trained on the pre-rename `*_adaptive_uniform_ratio` flags).

    Returns ``(score + ratio/N) / (score.sum() + ratio)``. Unlike `_mix_uniform`,
    the effective uniform weight depends on the magnitude of ``score.sum()``:
    when failure rates are small (S << N), the floor dominates and the
    distribution is closer to uniform; when failures are large (S ~ N), the
    score dominates. This non-stationarity is exactly the original behavior.

    Falls back to uniform when the score is all-zero (S=0) — the additive
    floor ratio/N is still finite, so the renormalization yields uniform.
    """
    n = score.numel()
    weighted = score + ratio / n
    s = weighted.sum()
    if float(s) > 0.0:
        return weighted / s
    return torch.full_like(score, 1.0 / n)


def _adaptive_probs(score: torch.Tensor, prob: float, ratio: float | None) -> torch.Tensor:
    """Dispatch helper: if a legacy `ratio` is set (not None), use the old
    additive-floor formula; otherwise use the new mixture-probability form.
    Mutual exclusion enforced upstream in train_bones.py / env_cfg defaults.
    """
    if ratio is not None:
        return _additive_floor(score, float(ratio))
    return _mix_uniform(score, float(prob))


class MultiClipMotionCommandCategorized(MultiClipMotionCommand):
    """Multi-clip motion command with a per-env category index for adaptive
    sampling over (categories × clips × frames).

    The clip → category mapping is built once at init by applying a named
    categorizer to each clip's name. The per-env `category_idx` buffer is kept
    in sync with `clip_ids` whenever clips are (re)assigned.
    """

    cfg: "MultiClipMotionCommandCategorizedCfg"

    def __init__(self, cfg: "MultiClipMotionCommandCategorizedCfg", env: "ManagerBasedRLEnv"):
        # In latent_kmeans mode, derive num_categories from the centroids JSON
        # BEFORE loading the zarr; categories / include_motion_types are
        # ignored (the JSON's clip→cluster_id map is authoritative).
        if getattr(cfg, "categorizer_mode", "keyword") == "latent_kmeans":
            if not getattr(cfg, "latent_centroids_path", None):
                raise ValueError(
                    "categorizer_mode='latent_kmeans' requires latent_centroids_path "
                    "(path to the JSON from scripts/cluster_motion_latents.py)."
                )
            import json as _json
            with open(cfg.latent_centroids_path, "r") as _f:
                _cluster_data = _json.load(_f)
            cfg.num_categories = int(_cluster_data["k"])
            cfg.include_motion_types = None  # don't filter by keyword
        elif cfg.categories is not None and len(cfg.categories) > 0:
            # If the user gave a `categories` list (keyword mode), derive
            # include_motion_types from it BEFORE the parent loads the zarr
            # (so unmatched clips never get loaded). Auto-derive num_categories
            # from the list length too. Categories list takes precedence over
            # the legacy `categorizer` field.
            cfg.num_categories = len(cfg.categories)
            if not cfg.include_motion_types:
                cfg.include_motion_types = list(cfg.categories)

        super().__init__(cfg, env)

        if getattr(cfg, "categorizer_mode", "keyword") == "latent_kmeans":
            from whole_body_tracking.tasks.generalist.mdp.categorizers import (
                make_latent_kmeans_categorizer,
            )
            fn = make_latent_kmeans_categorizer(cfg.latent_centroids_path)
        elif cfg.categories is not None and len(cfg.categories) > 0:
            fn = make_priority_categorizer(cfg.categories)
        else:
            if cfg.categorizer not in CATEGORIZERS:
                raise ValueError(
                    f"Unknown categorizer {cfg.categorizer!r}. "
                    f"Registered: {sorted(CATEGORIZERS.keys())}. "
                    f"Or pass `categories=[...]` to build one dynamically."
                )
            fn = CATEGORIZERS[cfg.categorizer]

        # Build clip → category lookup over the post-filter clip set.
        cats: list[int] = []
        unmatched: list[str] = []
        for name in self.motion.clip_names:
            try:
                cats.append(int(fn(name)))
            except ValueError:
                unmatched.append(name)
                if cfg.unmatched == "raise":
                    pass  # collect all, raise once below
                else:
                    cats.append(int(cfg.unmatched_default))

        if unmatched and cfg.unmatched == "raise":
            preview = unmatched[:10]
            raise ValueError(
                f"[Generalist] {len(unmatched)} clip(s) unmatched by categorizer "
                f"{cfg.categorizer!r}. Examples: {preview}"
            )

        self.clip_to_category = torch.tensor(cats, dtype=torch.long, device=self.device)
        assert self.clip_to_category.numel() == self.motion.num_clips
        if cfg.num_categories is not None:
            assert int(self.clip_to_category.max().item()) < cfg.num_categories, (
                f"categorizer produced index >= num_categories={cfg.num_categories}"
            )

        # Per-env category index, kept in sync with clip_ids.
        self.category_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.category_idx[:] = self.clip_to_category[self.clip_ids]

        # Resolve human-readable names for downstream logging. Latent-kmeans
        # gets generic names; dynamic `categories` takes priority for keyword
        # mode; otherwise look up the named categorizer's registered names
        # (with fallback to cat_0, cat_1, ...).
        K = int(self.clip_to_category.max().item()) + 1
        K = max(K, cfg.num_categories or K)
        if getattr(cfg, "categorizer_mode", "keyword") == "latent_kmeans":
            self.category_names = [f"lk_{i}" for i in range(K)]
        elif cfg.categories is not None and len(cfg.categories) > 0:
            self.category_names = list(cfg.categories)
            if len(self.category_names) < K:
                self.category_names += [
                    f"cat_{i}" for i in range(len(self.category_names), K)
                ]
        else:
            self.category_names = names_for(cfg.categorizer, K)
        # Cache num_categories so the runner can read it off the command term
        # without going through the cfg layer.
        self.num_categories = K

        # Diagnostic: print category histogram so the user can eyeball bucketing.
        print(f"[Generalist] categorizer={cfg.categorizer}  num_categories={cfg.num_categories}  "
              f"sampling_mode={getattr(cfg, 'sampling_mode', 'frame_uniform')}")
        for ci in range(K):
            mask = self.clip_to_category == ci
            n_clips = int(mask.sum().item())
            members = [self.motion.clip_names[i] for i, m in enumerate(mask.tolist()) if m]
            label = self.category_names[ci] if ci < len(self.category_names) else f"cat_{ci}"
            print(f"  cat {ci} ({label}): {n_clips} clips  e.g. {members[:5]}")

        # ── Per-category clip pools for balanced sampling ──────────────────
        # Packed layout: clips for cat 0 first, then cat 1, etc.
        # `clips_by_cat_offsets[k]` is the start of cat k's slab in
        # `clips_by_cat_flat`; `clips_by_cat_sizes[k]` is its length.
        # Lets _balanced_sampling pick n clips with three vectorized gathers.
        clips_by_cat = [
            (self.clip_to_category == k).nonzero(as_tuple=True)[0]
            for k in range(K)
        ]
        sizes_list = [int(p.numel()) for p in clips_by_cat]
        empty_cats = [self.category_names[k] for k, s in enumerate(sizes_list) if s == 0]
        # Every cat-aware sampling mode breaks (or silently writes uninitialized
        # clip_ids) when a cat has zero clips. Raise for ALL cat-aware modes,
        # not just balanced, with a clear message.
        _CAT_AWARE_MODES = (
            "balanced", "cat_uniform_clip_adaptive", "cat_adaptive_clip_uniform",
            "cat_blend_clip_uniform", "cat_adaptive_clip_adaptive",
        )
        if empty_cats and getattr(cfg, "sampling_mode", "frame_uniform") in _CAT_AWARE_MODES:
            raise RuntimeError(
                f"Cat-aware sampling mode {cfg.sampling_mode!r} requires every "
                f"category to have >= 1 clip. Empty categories: {empty_cats}. "
                f"Check --categories alignment with the zarr (in latent_kmeans "
                f"mode, re-cluster with a smaller --k)."
            )
        self.clips_by_cat_flat = torch.cat(clips_by_cat) if clips_by_cat else torch.empty(0, dtype=torch.long, device=self.device)
        self.clips_by_cat_sizes = torch.tensor(sizes_list, dtype=torch.long, device=self.device)
        # Cumsum-of-prefix gives slab starts: [0, s0, s0+s1, ...]
        offsets_list = [0] + sizes_list[:-1]
        for i in range(1, len(offsets_list)):
            offsets_list[i] = offsets_list[i - 1] + sizes_list[i - 1]
        self.clips_by_cat_offsets = torch.tensor(offsets_list, dtype=torch.long, device=self.device)

        # ── Per-clip failure-rate tracker (for sampling_mode == "clip_adaptive") ──
        # Mirrors the bin-based scheme in MultiClipMotionCommand: _current_clip_failed
        # gets bincounted from failed envs on reset; _update_command blends it into
        # clip_failure_rate with the existing `adaptive_alpha` EMA every step.
        self.clip_failure_rate = torch.zeros(self.motion.num_clips, dtype=torch.float, device=self.device)
        self._current_clip_failed = torch.zeros(self.motion.num_clips, dtype=torch.float, device=self.device)

        # ── Per-category failure tracker (for cat_blend_clip_uniform) ──
        # Direct per-category EMA over termination counts (denser/stabler than
        # averaging per-clip rates when there are many clips). Folded each step
        # in _update_command with the same `adaptive_alpha` as the bin-based path.
        self.cat_failure_rate = torch.zeros(K, dtype=torch.float, device=self.device)
        self._current_cat_failed = torch.zeros(K, dtype=torch.float, device=self.device)

        # ── Flight tables (per-clip stance baseline + flight-frame index, for
        # foot-clearance rewards/terminations in mdp/jumps.py and flight-biased RSI).
        self.flight_margin = float(getattr(cfg, "flight_margin", 0.07))
        self.flight_rsi_ratio = float(getattr(cfg, "flight_rsi_ratio", 0.0))
        self._build_flight_tables()

        # Re-balance the initial clip assignment if in balanced mode. Parent's
        # __init__ called _assign_random_clips(all_envs) BEFORE clips_by_cat_*
        # existed, so that call fell through to frame-uniform sampling. Re-do
        # it now so the very first rollout step starts balanced rather than
        # converging only as episodes reset.
        if getattr(cfg, "sampling_mode", "frame_uniform") == "balanced":
            self._balanced_sampling(torch.arange(self.num_envs, device=self.device))
            self.category_idx[:] = self.clip_to_category[self.clip_ids]
        elif getattr(cfg, "sampling_mode", "frame_uniform") == "clip_adaptive":
            # Same rationale as balanced: parent's __init__ ran _assign_random_clips
            # before clip_failure_rate existed, so the initial assignment fell
            # through to the parent's path. Re-do it now via clip_adaptive (which
            # is effectively uniform at this point since clip_failure_rate is zero).
            self._clip_adaptive_sampling(torch.arange(self.num_envs, device=self.device))
            self.category_idx[:] = self.clip_to_category[self.clip_ids]
        elif getattr(cfg, "sampling_mode", "frame_uniform") == "cat_uniform_clip_adaptive":
            self._cat_uniform_clip_adaptive_sampling(torch.arange(self.num_envs, device=self.device))
            self.category_idx[:] = self.clip_to_category[self.clip_ids]
        elif getattr(cfg, "sampling_mode", "frame_uniform") == "cat_adaptive_clip_uniform":
            self._cat_adaptive_clip_uniform_sampling(torch.arange(self.num_envs, device=self.device))
            self.category_idx[:] = self.clip_to_category[self.clip_ids]
        elif getattr(cfg, "sampling_mode", "frame_uniform") == "cat_blend_clip_uniform":
            self._cat_blend_clip_uniform_sampling(torch.arange(self.num_envs, device=self.device))
            self.category_idx[:] = self.clip_to_category[self.clip_ids]
        elif getattr(cfg, "sampling_mode", "frame_uniform") == "cat_adaptive_clip_adaptive":
            self._cat_adaptive_clip_adaptive_sampling(torch.arange(self.num_envs, device=self.device))
            self.category_idx[:] = self.clip_to_category[self.clip_ids]

    # ── Balanced sampling ─────────────────────────────────────────────────
    # Two-stage uniform sampler: cat → clip → frame. Equalizes per-category
    # rollout occupancy regardless of clip count or length. Activated by
    # `cfg.sampling_mode == "balanced"`; dispatched from both
    # _assign_random_clips and _adaptive_sampling overrides below so it
    # fires regardless of the --sampling flag.

    def _balanced_sampling(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        device = self.device
        K = self.num_categories

        # Stage 1: uniform category per env.
        cats = torch.randint(0, K, (n,), device=device)

        # Stage 2: uniform clip within the sampled category.
        sizes = self.clips_by_cat_sizes[cats]                       # [n]
        offsets = self.clips_by_cat_offsets[cats]                   # [n]
        local_idx = (torch.rand(n, device=device) * sizes.float()).long()
        local_idx = torch.minimum(local_idx, sizes - 1)             # safety against fp rounding
        clip_ids = self.clips_by_cat_flat[offsets + local_idx]      # [n]

        # Stage 3: uniform frame within the picked clip.
        clip_starts = self.motion.clip_start_idx[clip_ids]
        clip_ends = self.motion.clip_end_idx[clip_ids]
        lens = (clip_ends - clip_starts).clamp(min=1)
        local_frame = (torch.rand(n, device=device) * lens.float()).long()
        local_frame = torch.minimum(local_frame, lens - 1)
        time_steps = clip_starts + local_frame

        # Commit to per-env state. Cache refresh matches parent paths.
        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = clip_starts
        self.clip_end[env_ids] = clip_ends
        self.time_steps[env_ids] = time_steps
        self._cache_current_frames()

    # Refresh category_idx wherever clip_ids changes. The parent updates
    # clip_ids in two places: _assign_random_clips and _adaptive_sampling.
    # Override both to add the category sync after the parent's work.
    #
    # Note: parent's __init__ calls _assign_random_clips before our subclass
    # has built clip_to_category / category_idx, so the hasattr guard makes
    # the first invocation a no-op. After our __init__ finishes, the buffers
    # exist and every subsequent call updates category_idx in lockstep.
    #
    # Balanced-mode dispatch: when cfg.sampling_mode == "balanced", both
    # paths redirect to _balanced_sampling, so the failure-driven adaptive
    # bins are bypassed (they're meaningless until every category has
    # proportional rollout coverage — the imbalance is what we're fixing).

    # ── Clip-adaptive sampling ────────────────────────────────────────────
    # Per-clip clipped-adaptive sampler. Same termination signal and EMA
    # mechanics as the bin-based adaptive, but the curriculum lives on clips
    # rather than frame bins. The additive-uniform-offset (parallel to
    # `adaptive_uniform_ratio`) ensures every clip retains a floor probability
    # — addresses the bin-based mode's tendency to over-concentrate on a few
    # hard regions and under-train everything else.

    def _clip_adaptive_sampling(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        device = self.device

        # Stage 0: attribute failures to OLD clip_ids before we overwrite them.
        # Mirrors the bin-based path's pre-overwrite read at line ~920. Guarded
        # because this method also runs during the init re-balance, when the
        # env's termination_manager hasn't been constructed yet — at that point
        # clip_failure_rate is all zeros so the sampling distribution
        # degenerates to uniform via the additive floor, which is what we want.
        if hasattr(self._env, "termination_manager"):
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if episode_failed.any():
                failed_clip_ids = self.clip_ids[env_ids][episode_failed]
                self._current_clip_failed[:] = torch.bincount(
                    failed_clip_ids, minlength=self.motion.num_clips
                ).float()

        # Stage 1: per-clip distribution = EMA failure score + uniform offset.
        # Same additive-offset trick as `adaptive_uniform_ratio` in the
        # bin-based path: floor scales self-adjustingly with total signal.
        probs = _adaptive_probs(
            self.clip_failure_rate,
            self.cfg.clip_uniform_prob,
            self.cfg.clip_adaptive_uniform_ratio,
        )

        # Stage 2: sample clips.
        clip_ids = torch.multinomial(probs, n, replacement=True)

        # Stage 3: uniform frame within the picked clip.
        clip_starts = self.motion.clip_start_idx[clip_ids]
        clip_ends = self.motion.clip_end_idx[clip_ids]
        lens = (clip_ends - clip_starts).clamp(min=1)
        local_frame = (torch.rand(n, device=device) * lens.float()).long()
        local_frame = torch.minimum(local_frame, lens - 1)
        time_steps = clip_starts + local_frame

        # Commit env state. Cache refresh matches parent paths.
        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = clip_starts
        self.clip_end[env_ids] = clip_ends
        self.time_steps[env_ids] = time_steps
        self._cache_current_frames()

        # Sampling metrics (over clips for clip_adaptive). Keys match the
        # bin-based path so wandb dashboards stay valid; the "bin" suffix
        # is a historical naming carryover, not a semantic claim.
        H = -(probs * (probs + 1e-12).log()).sum()
        H_norm = H / torch.log(torch.tensor(float(self.motion.num_clips), device=device))
        pmax, imax = probs.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / float(self.motion.num_clips)

    # ── cat_uniform_clip_adaptive sampling ────────────────────────────────
    # Stage 1 uniform over cats (1/K), stage 2 clipped-adaptive within cat
    # (using clip_failure_rate restricted to the cat's clips), stage 3 uniform
    # frame. Reuses clip_adaptive_uniform_ratio for the within-cat floor.

    def _cat_uniform_clip_adaptive_sampling(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        device = self.device
        K = self.num_categories

        # Attribute failures to OLD clip_ids before overwriting. Guarded so
        # the init re-balance path (called before termination_manager exists)
        # works — at init clip_failure_rate is zero, so sampling falls back to
        # the additive uniform floor.
        if hasattr(self._env, "termination_manager"):
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if episode_failed.any():
                failed_clip_ids = self.clip_ids[env_ids][episode_failed]
                self._current_clip_failed[:] = torch.bincount(
                    failed_clip_ids, minlength=self.motion.num_clips
                ).float()

        # Stage 1: uniform cat.
        cats = torch.randint(0, K, (n,), device=device)

        # Stage 2: within-cat clipped-adaptive over clips. Loop over cats (K
        # is small, typically <10). Each iteration samples for all envs that
        # picked that cat.
        clip_ids = torch.empty(n, dtype=torch.long, device=device)
        offsets_cpu = self.clips_by_cat_offsets.tolist()
        sizes_cpu = self.clips_by_cat_sizes.tolist()
        for k in range(K):
            mask = cats == k
            n_in_cat = int(mask.sum().item())
            if n_in_cat == 0:
                continue
            size_k = sizes_cpu[k]
            if size_k == 0:
                # Should be guarded at init time, but no-op defensively.
                continue
            cat_clips = self.clips_by_cat_flat[offsets_cpu[k]:offsets_cpu[k] + size_k]
            cat_rates = self.clip_failure_rate[cat_clips]
            cat_probs = _adaptive_probs(
                cat_rates,
                self.cfg.clip_uniform_prob,
                self.cfg.clip_adaptive_uniform_ratio,
            )
            sampled = torch.multinomial(cat_probs, n_in_cat, replacement=True)
            clip_ids[mask] = cat_clips[sampled]

        # Stage 3: uniform frame.
        clip_starts = self.motion.clip_start_idx[clip_ids]
        clip_ends = self.motion.clip_end_idx[clip_ids]
        lens = (clip_ends - clip_starts).clamp(min=1)
        local_frame = (torch.rand(n, device=device) * lens.float()).long()
        local_frame = torch.minimum(local_frame, lens - 1)
        time_steps = clip_starts + local_frame

        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = clip_starts
        self.clip_end[env_ids] = clip_ends
        self.time_steps[env_ids] = time_steps
        self._cache_current_frames()

        # Metrics: cat marginal is uniform; log uniform stats so the keys
        # stay populated. (Within-cat shaping isn't summarized here.)
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / float(K)
        self.metrics["sampling_top1_bin"][:] = 0.0

    # ── cat_adaptive_clip_uniform sampling ────────────────────────────────
    # Stage 1 clipped-adaptive over cats (per-cat score = mean of
    # clip_failure_rate over that cat's clips, plus cat_adaptive_uniform_ratio
    # floor), stage 2 uniform clip within cat, stage 3 uniform frame.

    def _cat_adaptive_clip_uniform_sampling(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        device = self.device
        K = self.num_categories

        # Attribute failures to OLD clip_ids before overwriting. Guarded so
        # the init re-balance path (called before termination_manager exists)
        # works — at init clip_failure_rate is zero, so sampling falls back to
        # the additive uniform floor.
        if hasattr(self._env, "termination_manager"):
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if episode_failed.any():
                failed_clip_ids = self.clip_ids[env_ids][episode_failed]
                self._current_clip_failed[:] = torch.bincount(
                    failed_clip_ids, minlength=self.motion.num_clips
                ).float()

        # Stage 1: adaptive cat. Per-cat score = mean of clip_failure_rate.
        cat_sum = torch.zeros(K, dtype=torch.float, device=device)
        cat_sum.scatter_add_(0, self.clip_to_category, self.clip_failure_rate)
        cat_rates = cat_sum / self.clips_by_cat_sizes.float().clamp(min=1)
        cat_probs = _adaptive_probs(
            cat_rates,
            self.cfg.cat_uniform_prob,
            self.cfg.cat_adaptive_uniform_ratio,
        )
        cats = torch.multinomial(cat_probs, n, replacement=True)

        # Stage 2: uniform clip within cat (mirrors _balanced_sampling).
        sizes = self.clips_by_cat_sizes[cats]
        offsets = self.clips_by_cat_offsets[cats]
        local_idx = (torch.rand(n, device=device) * sizes.float()).long()
        local_idx = torch.minimum(local_idx, sizes - 1)
        clip_ids = self.clips_by_cat_flat[offsets + local_idx]

        # Stage 3: uniform frame.
        clip_starts = self.motion.clip_start_idx[clip_ids]
        clip_ends = self.motion.clip_end_idx[clip_ids]
        lens = (clip_ends - clip_starts).clamp(min=1)
        local_frame = (torch.rand(n, device=device) * lens.float()).long()
        local_frame = torch.minimum(local_frame, lens - 1)
        time_steps = clip_starts + local_frame

        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = clip_starts
        self.clip_end[env_ids] = clip_ends
        self.time_steps[env_ids] = time_steps
        self._cache_current_frames()

        # Metrics: log cat-level entropy / top-1.
        H = -(cat_probs * (cat_probs + 1e-12).log()).sum()
        H_norm = H / torch.log(torch.tensor(float(K), device=device))
        pmax, imax = cat_probs.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / float(K)

    # ── cat_blend_clip_uniform sampling ───────────────────────────────────
    # Stage 1 adaptive over cats using a DIRECT per-category EMA of termination
    # failures (cat_failure_rate), stage 2 uniform clip within cat, stage 3
    # uniform frame. Differs from cat_adaptive_clip_uniform in that the per-cat
    # score is a direct per-cat EMA rather than the mean of per-clip rates —
    # denser/stabler when there are many clips per cat.

    def _cat_blend_clip_uniform_sampling(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        device = self.device
        K = self.num_categories

        # Attribute failures to OLD categories before clip_ids is overwritten.
        # Guarded so the init re-balance (no termination_manager yet) is uniform.
        if hasattr(self._env, "termination_manager"):
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if episode_failed.any():
                failed_cats = self.category_idx[env_ids][episode_failed]
                self._current_cat_failed[:] = torch.bincount(failed_cats, minlength=K).float()

        # Stage 1: per-cat score = normalized cat_failure_rate + uniform floor.
        # Mix the per-cat failure distribution with uniform; cat_uniform_prob is
        # the probability mass on uniform (0 = pure failure-adaptive, 1 = uniform).
        fail_n = self.cat_failure_rate / self.cat_failure_rate.mean().clamp(min=1e-8)
        cat_probs = _adaptive_probs(
            fail_n,
            self.cfg.cat_uniform_prob,
            self.cfg.cat_adaptive_uniform_ratio,
        )
        cats = torch.multinomial(cat_probs, n, replacement=True)

        # Stage 2: uniform clip within cat (mirrors _cat_adaptive_clip_uniform).
        sizes = self.clips_by_cat_sizes[cats]
        offsets = self.clips_by_cat_offsets[cats]
        local_idx = torch.minimum((torch.rand(n, device=device) * sizes.float()).long(), sizes - 1)
        clip_ids = self.clips_by_cat_flat[offsets + local_idx]

        # Stage 3: uniform frame within clip.
        clip_starts = self.motion.clip_start_idx[clip_ids]
        clip_ends = self.motion.clip_end_idx[clip_ids]
        lens = (clip_ends - clip_starts).clamp(min=1)
        local_frame = torch.minimum((torch.rand(n, device=device) * lens.float()).long(), lens - 1)
        time_steps = clip_starts + local_frame

        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = clip_starts
        self.clip_end[env_ids] = clip_ends
        self.time_steps[env_ids] = time_steps
        self._cache_current_frames()

        # Cat-level sampling metrics.
        H = -(cat_probs * (cat_probs + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = H / torch.log(torch.tensor(float(K), device=device))
        pmax, imax = cat_probs.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / float(K)

    # ── cat_adaptive_clip_adaptive sampling ───────────────────────────────
    # Three-level adaptive sampling used by the generalist task.
    #   Stage 1 (cat):   per-cat score = cat_failure_rate / mean(cat_failure_rate)
    #                    + cat_adaptive_uniform_ratio/K floor.  Heavier adaptive.
    #   Stage 2 (clip):  per-clip score = clip_failure_rate[cat_clips] /
    #                    mean(clip_failure_rate[cat_clips]) +
    #                    clip_adaptive_uniform_ratio/|cat| floor. Lighter adaptive.
    #   Stage 3 (frame): uniform within clip.
    # Differs from cat_uniform_clip_adaptive (Stage 1 is uniform) and from
    # cat_adaptive_clip_uniform (Stage 2 is uniform) by making BOTH stages
    # adaptive, with different uniform floors per stage.

    def _cat_adaptive_clip_adaptive_sampling(self, env_ids: torch.Tensor):
        n = env_ids.numel()
        device = self.device
        K = self.num_categories

        # Attribute failures to OLD clip_ids and OLD categories before clip_ids
        # is overwritten. Both EMAs (per-clip and per-cat) are updated each
        # step in _update_command using these buffers.
        if hasattr(self._env, "termination_manager"):
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if episode_failed.any():
                failed_clip_ids = self.clip_ids[env_ids][episode_failed]
                self._current_clip_failed[:] = torch.bincount(
                    failed_clip_ids, minlength=self.motion.num_clips
                ).float()
                if hasattr(self, "_current_cat_failed"):
                    failed_cats = self.category_idx[env_ids][episode_failed]
                    self._current_cat_failed[:] = torch.bincount(failed_cats, minlength=K).float()

        # Stage 1: per-cat score = direct per-cat EMA, mixed with uniform by
        # cat_uniform_prob (probability mass on uniform; 0 = pure adaptive).
        fail_n = self.cat_failure_rate / self.cat_failure_rate.mean().clamp(min=1e-8)
        cat_probs = _adaptive_probs(
            fail_n,
            self.cfg.cat_uniform_prob,
            self.cfg.cat_adaptive_uniform_ratio,
        )
        cats = torch.multinomial(cat_probs, n, replacement=True)

        # Stage 2: within-cat clipped-adaptive over clips. Loop over cats (K
        # is small, typically <10). For each cat, sample clips for the envs
        # that picked that cat using the per-clip EMA + light uniform floor.
        clip_ids = torch.empty(n, dtype=torch.long, device=device)
        offsets_cpu = self.clips_by_cat_offsets.tolist()
        sizes_cpu = self.clips_by_cat_sizes.tolist()
        for k in range(K):
            mask = cats == k
            n_in_cat = int(mask.sum().item())
            if n_in_cat == 0:
                continue
            size_k = sizes_cpu[k]
            if size_k == 0:
                continue
            cat_clips = self.clips_by_cat_flat[offsets_cpu[k]:offsets_cpu[k] + size_k]
            cat_rates = self.clip_failure_rate[cat_clips]
            # Normalize per-clip rates by their own mean (so the additive floor
            # has stable relative weight), then add a HIGH floor for lighter
            # adaptive (default clip_adaptive_uniform_ratio = 0.6).
            cat_rates_n = cat_rates / cat_rates.mean().clamp(min=1e-8)
            cat_clip_probs = _adaptive_probs(
                cat_rates_n,
                self.cfg.clip_uniform_prob,
                self.cfg.clip_adaptive_uniform_ratio,
            )
            sampled = torch.multinomial(cat_clip_probs, n_in_cat, replacement=True)
            clip_ids[mask] = cat_clips[sampled]

        # Stage 3: uniform frame.
        clip_starts = self.motion.clip_start_idx[clip_ids]
        clip_ends = self.motion.clip_end_idx[clip_ids]
        lens = (clip_ends - clip_starts).clamp(min=1)
        local_frame = torch.minimum((torch.rand(n, device=device) * lens.float()).long(), lens - 1)
        time_steps = clip_starts + local_frame

        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = clip_starts
        self.clip_end[env_ids] = clip_ends
        self.time_steps[env_ids] = time_steps
        self._cache_current_frames()

        # Cat-level sampling metrics (same fields as the other cat-aware modes).
        H = -(cat_probs * (cat_probs + 1e-12).log()).sum()
        self.metrics["sampling_entropy"][:] = H / torch.log(torch.tensor(float(K), device=device))
        pmax, imax = cat_probs.max(dim=0)
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / float(K)

    # ── Flight calibration + flight-biased RSI ────────────────────────────

    def _build_flight_tables(self):
        """Precompute, from the reference motion, the per-clip stance baseline
        and a flat list of flight-frame indices grouped by clip. A frame is a
        flight frame when BOTH feet are above the clip's stance baseline by
        `flight_margin`. Used by the foot-clearance terms (jumps.py) and RSI."""
        foot_local = [i for i, name in enumerate(self.cfg.body_names) if name in ANKLE_NAMES]
        device = self.device
        F = self.motion.time_step_total
        nclips = self.motion.num_clips
        if len(foot_local) < 2:
            # Feet not tracked → no flight info; tables stay empty/zero.
            self.clip_foot_baseline = torch.zeros(nclips, device=device)
            self._flight_counts = torch.zeros(nclips, dtype=torch.long, device=device)
            self._flight_offsets = torch.zeros(nclips + 1, dtype=torch.long, device=device)
            self._flight_frames_flat = torch.zeros(0, dtype=torch.long, device=device)
            return

        foot_z = self.motion.body_pos_w[:, foot_local, 2]  # (F, 2) height above motion ground
        min_foot_z = foot_z.min(dim=-1).values  # (F,)

        # frame → clip id, marking frames outside any kept clip as invalid.
        frames = torch.arange(F, device=device)
        fc = torch.searchsorted(self.motion.clip_end_idx, frames, right=True).clamp(max=nclips - 1)
        valid = (frames >= self.motion.clip_start_idx[fc]) & (frames < self.motion.clip_end_idx[fc])

        # Per-clip stance baseline = min foot height over the clip's frames.
        baseline = torch.full((nclips,), float("inf"), device=device)
        baseline.scatter_reduce_(0, fc[valid], min_foot_z[valid], reduce="amin", include_self=True)
        baseline[torch.isinf(baseline)] = 0.0
        self.clip_foot_baseline = baseline

        # Flight frames (global ids, ascending → grouped by clip since clips are
        # contiguous ascending ranges), with per-clip counts/offsets (CSR).
        flight = valid & (min_foot_z > baseline[fc] + self.flight_margin)
        flight_idx = flight.nonzero(as_tuple=True)[0]
        counts = torch.bincount(fc[flight_idx], minlength=nclips)
        self._flight_counts = counts
        self._flight_offsets = torch.cat(
            [torch.zeros(1, dtype=torch.long, device=device), counts.cumsum(0)]
        )
        self._flight_frames_flat = flight_idx
        print(
            f"[Generalist] flight tables: {int(flight.sum())}/{F} flight frames across "
            f"{int((counts > 0).sum())}/{nclips} clips (margin={self.flight_margin} m)"
        )

    def _apply_flight_rsi(self, env_ids: torch.Tensor):
        """Flight-biased Reference State Init: with prob `flight_rsi_ratio`, move
        a reset's start frame to a random flight frame of its clip (clips with a
        flight phase only). RSI writes the reference pose+velocity at that frame,
        so the robot spawns airborne with the reference COM velocity."""
        # `_flight_counts` and `flight_rsi_ratio` are both set late in __init__
        # (after super()'s initial clip assignment), so check existence first to
        # avoid touching `flight_rsi_ratio` during that early call.
        if not hasattr(self, "_flight_counts") or self.flight_rsi_ratio <= 0.0:
            return
        if not hasattr(self._env, "termination_manager"):  # skip init-time assignment
            return
        cnt = self._flight_counts[self.clip_ids[env_ids]]
        pick = (torch.rand(env_ids.numel(), device=self.device) < self.flight_rsi_ratio) & (cnt > 0)
        if not pick.any():
            return
        sel = env_ids[pick]
        csel = self.clip_ids[sel]
        cnts = self._flight_counts[csel]
        offs = self._flight_offsets[csel]
        j = torch.minimum((torch.rand(sel.numel(), device=self.device) * cnts.float()).long(), cnts - 1)
        self.time_steps[sel] = self._flight_frames_flat[offs + j]
        self._cache_current_frames()

    def _assign_random_clips(self, env_ids: torch.Tensor):
        mode = getattr(self.cfg, "sampling_mode", "frame_uniform")
        if mode == "balanced" and hasattr(self, "clips_by_cat_flat"):
            self._balanced_sampling(env_ids)
        elif mode == "clip_adaptive" and hasattr(self, "clip_failure_rate"):
            self._clip_adaptive_sampling(env_ids)
        elif mode == "cat_uniform_clip_adaptive" and hasattr(self, "clip_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_uniform_clip_adaptive_sampling(env_ids)
        elif mode == "cat_adaptive_clip_uniform" and hasattr(self, "clip_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_adaptive_clip_uniform_sampling(env_ids)
        elif mode == "cat_blend_clip_uniform" and hasattr(self, "cat_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_blend_clip_uniform_sampling(env_ids)
        elif mode == "cat_adaptive_clip_adaptive" and hasattr(self, "cat_failure_rate") and hasattr(self, "clip_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_adaptive_clip_adaptive_sampling(env_ids)
        else:
            super()._assign_random_clips(env_ids)
        if hasattr(self, "category_idx"):
            self.category_idx[env_ids] = self.clip_to_category[self.clip_ids[env_ids]]
        self._apply_flight_rsi(env_ids)

    def _adaptive_sampling(self, env_ids: torch.Tensor):
        mode = getattr(self.cfg, "sampling_mode", "frame_uniform")
        if mode == "balanced" and hasattr(self, "clips_by_cat_flat"):
            self._balanced_sampling(env_ids)
        elif mode == "clip_adaptive" and hasattr(self, "clip_failure_rate"):
            self._clip_adaptive_sampling(env_ids)
        elif mode == "cat_uniform_clip_adaptive" and hasattr(self, "clip_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_uniform_clip_adaptive_sampling(env_ids)
        elif mode == "cat_adaptive_clip_uniform" and hasattr(self, "clip_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_adaptive_clip_uniform_sampling(env_ids)
        elif mode == "cat_blend_clip_uniform" and hasattr(self, "cat_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_blend_clip_uniform_sampling(env_ids)
        elif mode == "cat_adaptive_clip_adaptive" and hasattr(self, "cat_failure_rate") and hasattr(self, "clip_failure_rate") and hasattr(self, "clips_by_cat_flat"):
            self._cat_adaptive_clip_adaptive_sampling(env_ids)
        else:
            super()._adaptive_sampling(env_ids)
        if hasattr(self, "category_idx"):
            self.category_idx[env_ids] = self.clip_to_category[self.clip_ids[env_ids]]
        self._apply_flight_rsi(env_ids)

    def _update_command(self):
        # Parent does the time_steps tick, reset dispatch, cache refresh, and
        # the bin-based EMA. We piggyback the per-clip EMA update with the
        # exact same alpha so the two curriculum signals stay in lockstep.
        super()._update_command()
        a = self.cfg.adaptive_alpha
        self.clip_failure_rate = a * self._current_clip_failed + (1 - a) * self.clip_failure_rate
        self._current_clip_failed.zero_()

        # Per-category failure EMA (same alpha) for cat_blend_clip_uniform sampling.
        if hasattr(self, "cat_failure_rate"):
            self.cat_failure_rate = a * self._current_cat_failed + (1 - a) * self.cat_failure_rate
            self._current_cat_failed.zero_()


@configclass
class MultiClipMotionCommandCategorizedCfg(MultiClipMotionCommandCfg):
    """Configuration for the categorized multi-clip motion command."""

    class_type: type = MultiClipMotionCommandCategorized

    categories: list[str] | None = None
    """Primary API: comma-separated category names. When set, num_categories is
    derived as `len(categories)`, `include_motion_types` is auto-set to this
    same list (unless explicitly set), and the categorizer is built dynamically
    via `make_priority_categorizer(categories)`. List order = matching priority
    (first match wins), so put more-specific names first
    (e.g. `["stand_up", "walk"]` so `stand_up_to_walk` lands in stand_up).
    Takes precedence over `categorizer` and `num_categories` below."""

    categorizer: str = "walk_vs_standup"
    """Legacy API: name of a function in `generalist.mdp.categorizers.CATEGORIZERS`.
    Used only when `categories` is None."""

    categorizer_mode: str = "keyword"
    """How to derive per-clip categories.
    - 'keyword' (default): use `categories=[...]` keywords or the named
      `categorizer` function from CATEGORIZERS.
    - 'latent_kmeans': use the K-means centroids built from VAE latents
      (Phase 5). `latent_centroids_path` must be set to the JSON output of
      `scripts/cluster_motion_latents.py`. When this mode is selected, the
      `categories` and `categorizer` fields are ignored; `num_categories` is
      derived from the JSON's `k` field."""

    latent_centroids_path: str | None = None
    """Path to the latent-kmeans clusters JSON when `categorizer_mode ==
    'latent_kmeans'`. Ignored otherwise."""

    num_categories: int | None = 2
    """Legacy API: number of categories expected from the named `categorizer`.
    Auto-derived when `categories` is set."""

    unmatched: str = "raise"
    """How to handle clips the categorizer doesn't classify: 'raise' (default,
    fails fast — recommended in training) or 'default' (use unmatched_default).
    With the dynamic `categories` API, unmatched clips are normally filtered
    out at zarr-load time before the categorizer ever sees them."""

    unmatched_default: int = 0
    """Category index used when unmatched=='default'. Ignored otherwise."""

    sampling_mode: str = "frame_uniform"
    """Clip selection strategy on reset.
    - 'frame_uniform' (default): sample uniformly over the global frame
      timeline (clip-length-weighted; matches parent MultiClipMotionCommand
      behavior).
    - 'balanced': sample category uniformly (1/K), then clip uniformly
      within that category, then frame uniformly within the clip.
      Equalizes per-category rollout exposure regardless of clip count or
      length. Directly addresses the imbalance documented in
      markdowns/popart_implementation.md (Post-mortem). When set,
      overrides the per-rollout --sampling flag (adaptive vs uniform) for
      this command term — there's no within-category curriculum yet.
    - 'clip_adaptive': per-clip clipped-adaptive sampling. Builds a
      categorical distribution over clips from a per-clip EMA of failure
      counts (same termination signal as the bin-based adaptive, same
      `adaptive_alpha` EMA rate), with an additive uniform floor of
      `clip_adaptive_uniform_ratio / num_clips` per clip so no clip can
      starve. Then samples a frame uniformly within the chosen clip.
      Overrides the per-rollout --sampling flag like 'balanced' does.
    - 'cat_uniform_clip_adaptive': uniform category (1/K), then within the
      chosen category, clipped-adaptive over that category's clips using
      `clip_failure_rate` restricted to those clips with the same
      `clip_adaptive_uniform_ratio` floor, then uniform frame.
    - 'cat_adaptive_clip_uniform': adaptive over categories (per-cat score =
      mean of `clip_failure_rate` over that cat's clips, with additive
      `cat_adaptive_uniform_ratio` floor), then uniform clip within the cat,
      then uniform frame.
    - 'cat_blend_clip_uniform': adaptive over categories using a DIRECT per-cat
      termination-failure EMA (cat_failure_rate, normalized by its mean) with
      the same `cat_adaptive_uniform_ratio` floor, then uniform clip, then
      uniform frame. Differs from 'cat_adaptive_clip_uniform' in tracking the
      per-cat EMA directly rather than averaging per-clip rates.
    - 'cat_adaptive_clip_adaptive' (PRIMARY for generalist task): three-level
      adaptive sampler. Stage 1 cat-adaptive via direct per-cat EMA + low
      uniform floor (`cat_adaptive_uniform_ratio`, recommended 0.3 — heavier
      adaptive). Stage 2 within-cat clip-adaptive via per-clip EMA restricted
      to the cat's clips + HIGH uniform floor (`clip_adaptive_uniform_ratio`,
      recommended 0.6 — lighter adaptive). Stage 3 uniform frame. The per-
      stage floor asymmetry encodes: focus cat-level gradient share on
      categories the policy is failing, but inside each category keep
      coverage relatively even so we don't over-fit to a few specific clips."""

    flight_rsi_ratio: float = 0.0
    """Flight-biased Reference State Init. Fraction of resets whose clip has a
    flight phase that start mid-flight (airborne, with the reference COM
    velocity) instead of from a uniform frame. 0 = off. Self-gates: clips with
    no flight frames are unaffected."""

    flight_margin: float = 0.07
    """Height (m) a foot must clear its per-clip stance baseline to count as
    airborne. With the both-feet 'flight' test, this mainly rejects normal
    walking swing-foot lift; jumps clear it easily. ~0.07 sits above typical
    swing-foot height so flight is only flagged during genuine double-support-
    free phases."""

    clip_uniform_prob: float = 0.5
    """Probability of UNIFORM sampling at the CLIP stage (vs. failure-adaptive).
    The per-clip distribution is `(1 - p)·(failure / failure.sum()) + p·(1/N)`
    where N is the number of clips in the support. p=0 → pure adaptive, p=1 →
    fully uniform. Bounded [0, 1] and independent of N. Default 0.5.

    Used by `clip_adaptive`, `cat_uniform_clip_adaptive` and
    `cat_adaptive_clip_adaptive`. In the cat-aware modes the support is the
    current category's clips rather than all clips.

    (Renamed from the old additive-floor `clip_adaptive_uniform_ratio`; relation
    is p = ratio/(N+ratio). See `_mix_uniform`.)"""

    cat_uniform_prob: float = 0.5
    """Probability of UNIFORM sampling at the CATEGORY stage (vs. failure-adaptive).
    The per-cat distribution is `(1 - p)·(failure / failure.sum()) + p·(1/K)`
    where K = num_categories. p=0 → concentrate on the worst categories, p=1 →
    sample categories uniformly. Bounded [0, 1] and independent of K. Default 0.5.

    Used by `cat_adaptive_clip_uniform`, `cat_blend_clip_uniform` and
    `cat_adaptive_clip_adaptive`.

    (Renamed from the old additive-floor `cat_adaptive_uniform_ratio`; relation
    is p = ratio/(K+ratio). See `_mix_uniform`.)"""

    clip_adaptive_uniform_ratio: float | None = None
    """LEGACY additive-floor weight at the CLIP stage. When set (not None),
    overrides `clip_uniform_prob` and selects the pre-rename sampling formula
    ``probs = (score + ratio/N) / (score.sum() + ratio)`` — see `_additive_floor`.
    Default None → use `clip_uniform_prob` (new mixture form).

    Only restored so old checkpoints (run names tagged `clipRatio*`) can be
    resumed with the exact original sampling distribution; `*_uniform_prob` is
    the form to use for new runs."""

    cat_adaptive_uniform_ratio: float | None = None
    """LEGACY additive-floor weight at the CATEGORY stage. When set (not None),
    overrides `cat_uniform_prob` and selects the pre-rename sampling formula
    ``probs = (score + ratio/K) / (score.sum() + ratio)`` — see `_additive_floor`.
    Default None → use `cat_uniform_prob` (new mixture form).

    Only restored so old checkpoints (run names tagged `catRatio*`) can be
    resumed with the exact original sampling distribution; `*_uniform_prob` is
    the form to use for new runs."""