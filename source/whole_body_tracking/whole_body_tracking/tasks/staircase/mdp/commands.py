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
        
        # Store box position
        self.box_position = torch.tensor(self.cfg.box_position, device=self.device)
        self.box_rotation = torch.tensor(self.cfg.box_rotation, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )

        # VR 3-point related (ankle + pelvis tracking points)
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

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        pos = self.motion.joint_pos[self.time_steps]
        # Truncate to match robot DOF if necessary (e.g. 36 -> 29)
        robot_dof = self.robot.data.soft_joint_pos_limits.shape[1]
        if pos.shape[1] > robot_dof:
            return pos[:, :robot_dof]
        return pos

    @property
    def joint_vel(self) -> torch.Tensor:
        vel = self.motion.joint_vel[self.time_steps]
        # Truncate to match robot DOF if necessary
        robot_dof = self.robot.data.soft_joint_pos_limits.shape[1]
        if vel.shape[1] > robot_dof:
            return vel[:, :robot_dof]
        return vel

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

    @property
    def object_pos_w(self) -> torch.Tensor:
        """Position of the object in world frame."""
        return self._env.scene.env_origins + self.box_position

    @property
    def object_quat_w(self) -> torch.Tensor:
        """Orientation of the object in world frame."""
        return self.box_rotation

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        """Linear velocity of the object in world frame."""
        return torch.zeros_like(self.object_pos_w)

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        """Angular velocity of the object in world frame."""
        return torch.zeros_like(self.object_pos_w)

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
        eps_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        self.time_steps[env_ids[eps_mask]] = 0
        
        # Can add stride sampling to avoid near identical samples
        
        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        # Adjust robot position to be relative to the configured box position
        # Maintains the relative vector: (Robot - Box)_sim = (Robot - Box)_motion
        # root_pos[env_ids] += (self.box_position + self._env.scene.env_origins[env_ids]) - self.object_pos_w[env_ids]

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

        # Handle size mismatch (e.g. 36 in motion vs 29 in robot)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        robot_dof = soft_joint_pos_limits.shape[1]
        
        if joint_pos.shape[1] > robot_dof:
            joint_pos = joint_pos[:, :robot_dof]
            joint_vel = joint_vel[:, :robot_dof]

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
                # for name in self.cfg.body_names:
                #     self.current_body_visualizers.append(
                #         VisualizationMarkers(
                #             self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                #         )
                #     )
                #     self.goal_body_visualizers.append(
                #         VisualizationMarkers(
                #             self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                #         )
                #     )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            # for i in range(len(self.cfg.body_names)):
            #     self.current_body_visualizers[i].set_visibility(True)
            #     self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                # for i in range(len(self.cfg.body_names)):
                #     self.current_body_visualizers[i].set_visibility(False)
                #     self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        # for i in range(len(self.cfg.body_names)):
        #     self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
        #     self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    box_position: list[float] = [0.0, 0.0, 0.0]
    box_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    # sampling controls (safe defaults)
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


def _decode_zarr_strings(values) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


class StaircaseZarrMotionLoader:
    """Load multiclip staircase motions and per-clip staircase metadata from Zarr."""

    REQUIRED_KEYS = (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "clip_start_idx",
        "clip_end_idx",
        "clip_names",
        "staircase_id",
        "staircase_pos",
        "staircase_quat",
    )

    def __init__(self, zarr_path: str, body_indexes: Sequence[int], device: str = "cpu"):
        import zarr

        assert os.path.isdir(zarr_path), f"Invalid zarr path: {zarr_path}"
        store = zarr.open(zarr_path, mode="r")

        missing = [key for key in self.REQUIRED_KEYS if key not in store]
        if missing:
            raise KeyError(
                f"Staircase multiclip zarr is missing required datasets: {missing}. "
                f"Path: {zarr_path}"
            )

        self.fps = int(np.asarray(store["fps"])[0])
        self.clip_start_idx = torch.tensor(store["clip_start_idx"][:], dtype=torch.long)
        self.clip_end_idx = torch.tensor(store["clip_end_idx"][:], dtype=torch.long)
        self.num_clips = int(self.clip_start_idx.shape[0])
        self.clip_lengths = self.clip_end_idx - self.clip_start_idx
        self.clip_names = _decode_zarr_strings(store["clip_names"][:])

        self.staircase_id = torch.tensor(store["staircase_id"][:], dtype=torch.long, device=device)
        self.staircase_pos = torch.tensor(store["staircase_pos"][:], dtype=torch.float32, device=device)
        self.staircase_quat = torch.tensor(store["staircase_quat"][:], dtype=torch.float32, device=device)

        if self.staircase_pos.shape != (self.num_clips, 3):
            raise ValueError(
                f"Expected staircase_pos shape {(self.num_clips, 3)}, got {tuple(self.staircase_pos.shape)}"
            )
        if self.staircase_quat.shape != (self.num_clips, 4):
            raise ValueError(
                f"Expected staircase_quat shape {(self.num_clips, 4)}, got {tuple(self.staircase_quat.shape)}"
            )

        self.staircase_asset_path = (
            _decode_zarr_strings(store["staircase_asset_path"][:]) if "staircase_asset_path" in store else None
        )
        self.staircase_usd_dir = (
            _decode_zarr_strings(store["staircase_usd_dir"][:]) if "staircase_usd_dir" in store else None
        )
        self.staircase_raycast_asset_path = (
            _decode_zarr_strings(store["staircase_raycast_asset_path"][:])
            if "staircase_raycast_asset_path" in store
            else None
        )

        self.joint_pos = torch.tensor(store["joint_pos"][:], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(store["joint_vel"][:], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(store["body_pos_w"][:], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(store["body_quat_w"][:], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(store["body_lin_vel_w"][:], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(store["body_ang_vel_w"][:], dtype=torch.float32, device=device)

        self._body_indexes = (
            body_indexes.to(device)
            if isinstance(body_indexes, torch.Tensor)
            else torch.tensor(body_indexes, dtype=torch.long, device=device)
        )
        self.time_step_total = int(self.joint_pos.shape[0])

        print(
            f"[StaircaseZarrMotionLoader] Loaded {self.num_clips} clips, "
            f"{self.time_step_total} total frames @ {self.fps} fps, "
            f"{self._body_pos_w.shape[1]} bodies, device={device}"
        )

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


class StaircaseMultiClipMotionCommand(MotionCommand):
    """Staircase-specific multiclip motion command with per-clip staircase metadata."""

    cfg: "StaircaseMultiClipMotionCommandCfg"

    def __init__(self, cfg: "StaircaseMultiClipMotionCommandCfg", env: ManagerBasedRLEnv):
        CommandTerm.__init__(self, cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.min_sample_idx = cfg.min_sample_idx
        self.max_sample_idx = cfg.max_sample_idx
        self.steps_collect = cfg.steps_collect

        self.motion = StaircaseZarrMotionLoader(self.cfg.zarr_path, self.body_indexes, device=self.device)
        self.motion.clip_start_idx = self.motion.clip_start_idx.to(self.device)
        self.motion.clip_end_idx = self.motion.clip_end_idx.to(self.device)
        self.motion.clip_lengths = self.motion.clip_lengths.to(self.device)

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.clip_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_staircase_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_staircase_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.current_staircase_quat = torch.zeros(self.num_envs, 4, dtype=torch.float32, device=self.device)
        self.current_staircase_quat[:, 0] = 1.0
        self.current_clip_names = [""] * self.num_envs
        self._staircase_scene_dirty = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        self.total_frames = int(self.motion.clip_end_idx.max().item())
        self.bin_count = max(self.total_frames // 50, 100)
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.vr_3point_body_indices = [self.robot.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_indices_motion = [self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_offsets = torch.tensor(
            self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device
        ).view(1, -1, 3).repeat(self.num_envs, 1, 1)

        self.down_dir = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).view(1, -1, 3).repeat(
            self.num_envs, 1, 1
        )

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

        self._use_adaptive = os.environ.get("BONES_SAMPLING", "adaptive") == "adaptive"
        self._debug_prints_enabled = os.environ.get("WBT_STAIRCASE_MULTICLIP_DEBUG", "0") == "1"
        self._assign_clips(torch.arange(self.num_envs, device=self.device))

    @property
    def command(self) -> torch.Tensor:
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

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._env.scene.env_origins + self.current_staircase_pos

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.current_staircase_quat

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return torch.zeros_like(self.current_staircase_pos)

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return torch.zeros_like(self.current_staircase_pos)

    @property
    def clip_phase(self) -> torch.Tensor:
        progress = (self.time_steps - self.clip_start).float() / (self.clip_end - self.clip_start).float().clamp(min=1)
        return progress.unsqueeze(-1)

    def _refresh_staircase_metadata(self, env_ids: torch.Tensor):
        clip_ids = self.clip_ids[env_ids]
        self.current_staircase_id[env_ids] = self.motion.staircase_id[clip_ids]
        self.current_staircase_pos[env_ids] = self.motion.staircase_pos[clip_ids]
        self.current_staircase_quat[env_ids] = self.motion.staircase_quat[clip_ids]
        for env_id, clip_id in zip(env_ids.tolist(), clip_ids.tolist()):
            self.current_clip_names[env_id] = self.motion.clip_names[clip_id]
        self._staircase_scene_dirty[env_ids] = True

    def _debug_log_samples(self, env_ids: torch.Tensor):
        if not self._debug_prints_enabled or env_ids.numel() == 0:
            return
        env_id = int(env_ids[0].item())
        pos = self.current_staircase_pos[env_id].detach().cpu().tolist()
        quat = self.current_staircase_quat[env_id].detach().cpu().tolist()
        print(
            "[STAIRCASE_MULTICLIP_DEBUG] "
            f"env={env_id} clip_id={int(self.clip_ids[env_id])} "
            f"clip_name={self.current_clip_names[env_id]} "
            f"staircase_id={int(self.current_staircase_id[env_id])} "
            f"staircase_pos={pos} staircase_quat={quat} "
            f"timestep={int(self.time_steps[env_id])}"
        )

    def _assign_fixed_clip(self, env_ids: torch.Tensor):
        fixed_clip_index = int(self.cfg.fixed_clip_index)
        if fixed_clip_index < 0 or fixed_clip_index >= self.motion.num_clips:
            raise ValueError(
                f"fixed_clip_index={fixed_clip_index} is out of range for {self.motion.num_clips} clips."
            )

        clip_ids = torch.full((len(env_ids),), fixed_clip_index, dtype=torch.long, device=self.device)
        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = self.motion.clip_start_idx[clip_ids]
        self.clip_end[env_ids] = self.motion.clip_end_idx[clip_ids]
        self.time_steps[env_ids] = self.clip_start[env_ids]
        self._refresh_staircase_metadata(env_ids)
        self._debug_log_samples(env_ids)

    def _assign_random_clips(self, env_ids: torch.Tensor):
        n = len(env_ids)
        global_idx = torch.randint(0, self.total_frames, (n,), device=self.device)
        clip_ids = torch.searchsorted(self.motion.clip_end_idx, global_idx, right=True)
        clip_ids = torch.clamp(clip_ids, max=self.motion.num_clips - 1)
        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = self.motion.clip_start_idx[clip_ids]
        self.clip_end[env_ids] = self.motion.clip_end_idx[clip_ids]
        self.time_steps[env_ids] = global_idx.clamp(self.clip_start[env_ids], self.clip_end[env_ids] - 1)
        self._refresh_staircase_metadata(env_ids)
        self._debug_log_samples(env_ids)

    def _assign_clips(self, env_ids: torch.Tensor):
        if self.cfg.fixed_clip_index is not None:
            self._assign_fixed_clip(env_ids)
        else:
            self._assign_random_clips(env_ids)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        episode_failed = self._env.termination_manager.terminated[env_ids_t]
        if torch.any(episode_failed):
            fail_bins = torch.clamp(
                (self.time_steps[env_ids_t][episode_failed] * self.bin_count) // max(self.total_frames, 1),
                0,
                self.bin_count - 1,
            )
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count).float()

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

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids_t), replacement=True)
        global_frames = (
            (sampled_bins.float() + torch.rand(len(env_ids_t), device=self.device))
            / self.bin_count
            * self.total_frames
        ).long().clamp(0, self.total_frames - 1)

        clip_ids = torch.searchsorted(self.motion.clip_end_idx, global_frames, right=True)
        clip_ids = clip_ids.clamp(0, self.motion.num_clips - 1)
        self.clip_ids[env_ids_t] = clip_ids
        self.clip_start[env_ids_t] = self.motion.clip_start_idx[clip_ids]
        self.clip_end[env_ids_t] = self.motion.clip_end_idx[clip_ids]
        self.time_steps[env_ids_t] = global_frames.clamp(self.clip_start[env_ids_t], self.clip_end[env_ids_t] - 1)

        self._refresh_staircase_metadata(env_ids_t)
        self._debug_log_samples(env_ids_t)

        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        if self.cfg.fixed_clip_index is not None:
            self._assign_fixed_clip(env_ids_t)
        elif self._use_adaptive:
            self._adaptive_sampling(env_ids_t)
        else:
            self._assign_random_clips(env_ids_t)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_t), 6), device=self.device)
        root_pos[env_ids_t] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids_t] = quat_mul(orientations_delta, root_ori[env_ids_t])

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids_t), 6), device=self.device)
        root_lin_vel[env_ids_t] += rand_samples[:, :3]
        root_ang_vel[env_ids_t] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids_t]
        joint_pos[env_ids_t] = torch.clip(
            joint_pos[env_ids_t], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        self.robot.write_joint_state_to_sim(joint_pos[env_ids_t], joint_vel[env_ids_t], env_ids=env_ids_t)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids_t], root_ori[env_ids_t], root_lin_vel[env_ids_t], root_ang_vel[env_ids_t]], dim=-1),
            env_ids=env_ids_t,
        )

    def _update_command(self):
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.clip_end)[0]
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

    def sync_staircase_scene(self, scene):
        dirty_env_ids = torch.where(self._staircase_scene_dirty)[0]
        if dirty_env_ids.numel() == 0:
            return

        hidden_pos = torch.tensor(self.cfg.hidden_staircase_position, dtype=torch.float32, device=self.device)
        hidden_quat = torch.tensor(self.cfg.hidden_staircase_quat, dtype=torch.float32, device=self.device)

        for staircase_id, asset_name in self.cfg.staircase_variant_names.items():
            if asset_name not in scene.rigid_objects:
                raise KeyError(
                    f"Expected staircase rigid object '{asset_name}' for staircase_id={staircase_id}, "
                    f"available={sorted(scene.rigid_objects.keys())}"
                )
            asset = scene.rigid_objects[asset_name]
            state = asset.data.root_state_w.clone()
            active_env_ids = dirty_env_ids[self.current_staircase_id[dirty_env_ids] == staircase_id]
            inactive_env_ids = dirty_env_ids[self.current_staircase_id[dirty_env_ids] != staircase_id]

            if active_env_ids.numel() > 0:
                active_state = state[active_env_ids].clone()
                active_state[:, 0:3] = self.object_pos_w[active_env_ids]
                active_state[:, 3:7] = self.object_quat_w[active_env_ids]
                active_state[:, 7:13] = 0.0
                asset.write_root_state_to_sim(active_state, env_ids=active_env_ids)

            if inactive_env_ids.numel() > 0:
                inactive_state = state[inactive_env_ids].clone()
                inactive_state[:, 0:3] = scene.env_origins[inactive_env_ids] + hidden_pos
                inactive_state[:, 3:7] = hidden_quat
                inactive_state[:, 7:13] = 0.0
                asset.write_root_state_to_sim(inactive_state, env_ids=inactive_env_ids)

        self._staircase_scene_dirty[dirty_env_ids] = False


@configclass
class StaircaseMultiClipMotionCommandCfg(MotionCommandCfg):
    """Configuration for staircase multiclip motion command using a Zarr store."""

    class_type: type = StaircaseMultiClipMotionCommand

    motion_file: str = ""
    """Unused in multiclip mode. Overridden to satisfy config validation."""

    zarr_path: str = MISSING
    staircase_variant_names: dict[int, str] = MISSING
    fixed_clip_index: int | None = None
    hidden_staircase_position: list[float] = [0.0, 0.0, -50.0]
    hidden_staircase_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


def make_staircase_multiclip_motion_cfg(
    base_cfg: MotionCommandCfg,
    zarr_path: str,
    staircase_variant_names: dict[int, str],
) -> StaircaseMultiClipMotionCommandCfg:
    """Create a multiclip staircase command cfg from the existing single-motion cfg."""

    cfg = StaircaseMultiClipMotionCommandCfg(
        asset_name=base_cfg.asset_name,
        motion_file="",
        anchor_body_name=base_cfg.anchor_body_name,
        body_names=list(base_cfg.body_names),
        box_position=list(base_cfg.box_position),
        box_rotation=tuple(base_cfg.box_rotation),
        pose_range=dict(base_cfg.pose_range),
        velocity_range=dict(base_cfg.velocity_range),
        joint_position_range=tuple(base_cfg.joint_position_range),
        adaptive_kernel_size=base_cfg.adaptive_kernel_size,
        adaptive_lambda=base_cfg.adaptive_lambda,
        adaptive_uniform_ratio=base_cfg.adaptive_uniform_ratio,
        adaptive_alpha=base_cfg.adaptive_alpha,
        anchor_visualizer_cfg=base_cfg.anchor_visualizer_cfg,
        body_visualizer_cfg=base_cfg.body_visualizer_cfg,
        min_sample_idx=base_cfg.min_sample_idx,
        max_sample_idx=base_cfg.max_sample_idx,
        steps_collect=base_cfg.steps_collect,
        force_update_frequency=base_cfg.force_update_frequency,
        max_force=base_cfg.max_force,
        force_push_body=list(base_cfg.force_push_body),
        force_push_body_offset=[list(offset) for offset in base_cfg.force_push_body_offset],
        vr_3point_body=list(base_cfg.vr_3point_body),
        vr_3point_body_offset=[list(offset) for offset in base_cfg.vr_3point_body_offset],
        zarr_path=zarr_path,
        staircase_variant_names=dict(staircase_variant_names),
    )
    for attr in ("resampling_time_range", "debug_vis"):
        if hasattr(base_cfg, attr):
            setattr(cfg, attr, getattr(base_cfg, attr))
    return cfg
