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
        eps_mask = torch.rand(len(env_ids), device=self.device) < 0.1
        self.time_steps[env_ids[eps_mask]] = 0

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _uniform_sampling(self, env_ids: Sequence[int]):
        phase = sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device)
        time_samples = (phase * (self.motion.time_step_total - 1)).long()
        if self.max_sample_idx is not None and self.steps_collect is not None:
            sampling_range = (self.max_sample_idx - self.steps_collect) - self.min_sample_idx
            time_samples = (phase * sampling_range + self.min_sample_idx).long()
            time_samples = torch.clip(time_samples, min=self.min_sample_idx, max=self.max_sample_idx - self.steps_collect) #.long()

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
    adaptive_uniform_ratio: float = 0.1
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
                 exclude_props: list[str] | None = None, max_clips: int | None = None):
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

        # Limit number of clips if requested
        if max_clips is not None and max_clips < len(valid_indices):
            valid_indices = valid_indices[:max_clips]
            print(f"[ZarrMotionLoader] Limited to {max_clips} clips")

        self.clip_start_idx = torch.tensor([all_clip_start[i] for i in valid_indices], dtype=torch.long)
        self.clip_end_idx = torch.tensor([all_clip_end[i] for i in valid_indices], dtype=torch.long)
        self.num_clips = len(self.clip_start_idx)
        self.clip_lengths = self.clip_end_idx - self.clip_start_idx

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
        exclude_props = ["object manipulation"] if self.cfg.exclude_objects else None
        self.motion = ZarrMotionLoader(self.cfg.zarr_path, self.body_indexes, device=self.device,
                                       exclude_props=exclude_props,
                                       max_clips=self.cfg.max_clips)

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

        # Adaptive sampling: per-clip failure tracking
        self.bin_count = self.motion.num_clips
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
        """Assign random clips and random start frames to the given envs."""
        n = len(env_ids)
        clip_ids = torch.randint(0, self.motion.num_clips, (n,), device=self.device)
        self.clip_ids[env_ids] = clip_ids
        self.clip_start[env_ids] = self.motion.clip_start_idx[clip_ids]
        self.clip_end[env_ids] = self.motion.clip_end_idx[clip_ids]

        # Random start within each clip
        clip_lens = self.clip_end[env_ids] - self.clip_start[env_ids]
        offsets = (torch.rand(n, device=self.device) * (clip_lens - 1).float()).long()
        self.time_steps[env_ids] = self.clip_start[env_ids] + offsets

        # Refresh cache for newly assigned envs
        self._cache_current_frames()

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        env_ids_t = torch.as_tensor(env_ids, device=self.device)

        # Track failures per clip (adaptive sampling across clips)
        episode_failed = self._env.termination_manager.terminated[env_ids_t]
        if torch.any(episode_failed):
            fail_clips = self.clip_ids[env_ids_t][episode_failed]
            self._current_bin_failed[:] = torch.bincount(
                fail_clips, minlength=self.motion.num_clips
            ).float()

        # Assign new random clips
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

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Not used — clip sampling logic is in _resample_command."""
        pass


@configclass
class MultiClipMotionCommandCfg(MotionCommandCfg):
    """Configuration for multi-clip motion command using a Zarr store."""

    class_type: type = MultiClipMotionCommand

    zarr_path: str = MISSING
    """Path to the Zarr motion store."""

    exclude_objects: bool = True
    """Whether to exclude motions with object manipulation (content_props). Default: True."""

    max_clips: int | None = None
    """Maximum number of clips to load. None = load all. Useful for play/eval on smaller GPUs."""

    future_steps: list[int] = []
    """Future timestep offsets to include in observations (e.g., [5, 10, 15]).
    Empty list = no future frames cached. Each offset is clamped to clip boundaries."""

    # Override motion_file — not used in multi-clip mode
    motion_file: str = ""