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

class MultiMotionLoader:
    def __init__(self, motion_files: list[str], body_indexes: Sequence[int], has_wrist_grasp_label: bool, device: str = "cpu"):
        assert os.path.isfile(motion_files[0]), f"Invalid file path: {motion_files[0]}"
        try:
            datas = [np.load(motion_file) for motion_file in motion_files]
        except:
            breakpoint()
        self.time_step_total = sum([data["joint_pos"].shape[0] for data in datas])
        self.motion_start_idx = torch.tensor(([0] + [data["joint_pos"].shape[0] for data in datas])[:-1], dtype=torch.long, device=device).cumsum(dim=0)
        self.motion_end_idx = torch.tensor([data["joint_pos"].shape[0] for data in datas], dtype=torch.long, device=device).cumsum(dim=0)
        
        self.num_motions = len(datas)
        self.fps = datas[0]["fps"] # Assume all motion have same fps
        self.joint_pos = torch.tensor(np.concatenate([data["joint_pos"] for data in datas], axis=0), dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(np.concatenate([data["joint_vel"] for data in datas], axis=0), dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(np.concatenate([data["body_pos_w"] for data in datas], axis=0), dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(np.concatenate([data["body_quat_w"] for data in datas], axis=0), dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(np.concatenate([data["body_lin_vel_w"] for data in datas], axis=0), dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(np.concatenate([data["body_ang_vel_w"] for data in datas], axis=0), dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        if has_wrist_grasp_label:
            self._wrist_grasp_label = torch.tensor(np.concatenate([data["wrist_grasp_label"] for data in datas], axis=0), dtype=torch.bool, device=device)

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
    
    @property
    def wrist_grasp_label(self) -> torch.Tensor:
        if hasattr(self, '_wrist_grasp_label'):
            return self._wrist_grasp_label
        else:
            assert False, "No wrist grasp label in the motion data!"

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

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count #+ self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = (sampling_probabilities / sampling_probabilities.sum()) * (1-self.cfg.adaptive_uniform_ratio)
        sampling_probabilities += self.cfg.adaptive_uniform_ratio / float(self.bin_count) # correct implementation

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

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

class MultiMotionCommand(CommandTerm):
    cfg: MultiMotionCommandCfg

    def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.motion = MultiMotionLoader(self.cfg.motion_files, self.body_indexes, device=self.device, has_wrist_grasp_label=self.cfg.has_wrist_grasp_label)
        self.motion_ids = torch.randint(0, self.motion.num_motions, (self.num_envs,), device=self.device)
        #self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) # Should be random
        self.time_steps = self.motion.motion_start_idx[self.motion_ids]

        self.look_ahead_steps_init = (torch.arange(self.cfg.look_ahead_frames,device=self.device, dtype=torch.long)*self.cfg.look_ahead_frame_skips).view(1,-1).repeat(self.num_envs, 1)

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

        # Add vr 3point related terms
        self.vr_3point_body_indices = [self.robot.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_indices_motion = [self.cfg.body_names.index(name) for name in self.cfg.vr_3point_body]
        self.vr_3point_body_offsets = torch.tensor(self.cfg.vr_3point_body_offset, dtype=torch.float32, device=self.device).view(1,-1,3).repeat(self.num_envs, 1, 1)

        self.isaaclab_to_mujoco_dof= [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
        self.mujoco_to_isaaclab_dof= [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        self.lower_joint_indices_mujoco = list(range(12))
        self.lower_joint_isaaclab_indices = [self.isaaclab_to_mujoco_dof[i] for i in self.lower_joint_indices_mujoco]

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

        self.down_dir = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device).view(1,-1,3).repeat(self.num_envs, 1, 1) # [num_envs, 1, 3]

        # Force push related
        self.force_update_frequency = self.cfg.force_update_frequency
        self.max_force = self.cfg.max_force
        self.num_bodies = len(self.cfg.body_names)
        self.body_force_dir_buf = torch.randn(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.body_force_dir_buf /= torch.norm(self.body_force_dir_buf, dim=-1, keepdim=True) # normalize
        self.body_force_magnitude_buf = torch.rand(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False) # [0, 1]

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
        
        self.force_push_body_offsets = torch.tensor(self.cfg.force_push_body_offset, dtype=torch.float32, device=self.device).view(1,-1,3).repeat(self.num_envs, 1, 1)
        self.last_force_applied = torch.zeros(self.num_envs, len(self.force_push_ids), 3, dtype=torch.float, device=self.device, requires_grad=False)
        # Compliance related
        self.compliance_counter = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.compliance_duration_per_env = torch.zeros(self.num_envs, dtype=torch.int, device=self.device)
        self.eef_stiffness_buf = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.compliance_config_init = False

        self.metrics["force applied"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def command_lower_body(self) -> torch.Tensor:
        return torch.cat([self.joint_pos_lower_body, self.joint_vel_lower_body], dim=1)

    @property
    def command_lookahead(self) -> torch.Tensor:
        return torch.cat([self.joint_pos_lookahead, self.joint_vel_lookahead],dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def joint_pos_lower_body(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps][:, self.lower_joint_isaaclab_indices]
    
    @property
    def joint_vel_lower_body(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps][:, self.lower_joint_isaaclab_indices]

    @property
    def joint_pos_lookahead(self) -> torch.Tensor:
        return self.motion.joint_pos[self.future_time_steps].view(self.num_envs,-1)

    @property
    def joint_vel_lookahead(self) -> torch.Tensor:
        return self.motion.joint_vel[self.future_time_steps].view(self.num_envs,-1)

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
    def future_time_steps(self) -> torch.Tensor:
        return torch.clip(self.look_ahead_steps_init + self.time_steps[:,None], 
                          max=self.motion.motion_end_idx[self.motion_ids][:,None]-1).flatten().long()

    # VR 3 point related utils
    @property
    def vr_3point_body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps][:,self.vr_3point_body_indices_motion]
    
    @property
    def vr_3point_body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps][:,self.vr_3point_body_indices_motion] + quat_apply(self.vr_3point_body_quat_w, self.vr_3point_body_offsets)+ self._env.scene.env_origins[:, None, :]

    @property
    def robot_vr_3point_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:,self.vr_3point_body_indices]

    @property
    def robot_vr_3point_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.vr_3point_body_indices] + quat_apply(self.robot_vr_3point_quat_w, self.vr_3point_body_offsets)

    @property
    def wrist_grasp_label(self) -> torch.Tensor:
        return self.motion.wrist_grasp_label[self.time_steps]

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
        sampling_probabilities = self.bin_failed_count #+ self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = (sampling_probabilities / (sampling_probabilities.sum()+1e-8)) * (1-self.cfg.adaptive_uniform_ratio)
        sampling_probabilities += self.cfg.adaptive_uniform_ratio / float(self.bin_count) # correct implementation
        

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()
        
        # Determine motion_ids from time_steps using start/end indices
        # For each time_step, find which motion it belongs to by checking against motion boundaries
        motion_ids = torch.zeros_like(self.time_steps[env_ids], dtype=torch.long, device=self.device)
        for i in range(self.motion.num_motions):
            # Create mask for time_steps that fall within this motion's range
            start_idx = self.motion.motion_start_idx[i]
            end_idx = self.motion.motion_end_idx[i]
            mask = (self.time_steps[env_ids] >= start_idx) & (self.time_steps[env_ids] < end_idx)
            motion_ids[mask] = i
        
        self.motion_ids[env_ids] = motion_ids

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
        #env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        env_ids = torch.where(self.time_steps >= self.motion.motion_end_idx[self.motion_ids])[0]
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


class BindedMultiMotionCommand(MultiMotionCommand):
    """Multi-motion command where each environment is permanently bound to a motion ID.

    Each environment is assigned a motion via env_id % num_motions, so when there are
    more environments than motions, multiple envs share the same motion (wrapping around).
    Uses per-motion bins for adaptive sampling and records failures from all relevant envs.
    """

    cfg: BindedMultiMotionCommandCfg

    def __init__(self, cfg: BindedMultiMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # Override: bind each env to a motion ID using modulo (wrap around when num_envs > num_motions)
        self.motion_ids = (
            torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            % self.motion.num_motions
        )
        # Set initial time_steps to start of each env's bound motion
        self.time_steps = self.motion.motion_start_idx[self.motion_ids].clone()

        # Per-motion bins: each motion has its own bin structure
        steps_per_bin = 1.0 / (env.cfg.decimation * env.cfg.sim.dt)
        self.motion_lengths = (
            self.motion.motion_end_idx - self.motion.motion_start_idx
        )
        self.bin_count_per_motion = torch.clamp(
            (self.motion_lengths.float() / steps_per_bin).long() + 1,
            min=1,
        )
        self.max_bin_count = int(self.bin_count_per_motion.max().item())

        # Replace parent's single bin buffers with per-motion: (num_motions, max_bin_count)
        self.bin_failed_count = torch.zeros(
            self.motion.num_motions, self.max_bin_count, dtype=torch.float, device=self.device
        )
        self._current_bin_failed = torch.zeros(
            self.motion.num_motions, self.max_bin_count, dtype=torch.float, device=self.device
        )
        self.bin_count = self.max_bin_count  # keep for parent's kernel if needed

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        """Sample within each env's bound motion using per-motion adaptive bin sampling."""
        # Resolve env_ids (can be slice from reset, or list/tensor)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            env_ids = env_ids.flatten()
        n_envs = len(env_ids)

        # Record failures from ALL relevant envs (all envs being resampled that actually failed)
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            failed_env_ids = env_ids[episode_failed]
            bound_motion_ids = self.motion_ids[failed_env_ids]
            failed_time_steps = self.time_steps[failed_env_ids]

            starts = self.motion.motion_start_idx[bound_motion_ids]
            lengths = (
                self.motion.motion_end_idx[bound_motion_ids]
                - self.motion.motion_start_idx[bound_motion_ids]
            )
            lengths = torch.clamp(lengths, min=1)

            # Bin index within each motion: (time_step - start) * bin_count / length
            bin_counts = self.bin_count_per_motion[bound_motion_ids]
            rel_steps = (failed_time_steps - starts).float().clamp(min=0)
            bin_indices = (
                (rel_steps * bin_counts.float() / lengths.float()).long().clamp(max=bin_counts - 1)
            )

            # Vectorized scatter-add: linear_idx = motion_id * max_bin_count + bin_idx
            linear_indices = bound_motion_ids * self.max_bin_count + bin_indices
            ones = torch.ones(
                len(linear_indices), dtype=torch.float, device=self.device
            )
            self._current_bin_failed.view(-1).index_add_(0, linear_indices, ones)

        # Sample per env within its bound motion
        bound_motion_ids = self.motion_ids[env_ids]

        # Batched prob computation: conv1d + normalize for all motions at once
        probs_all = self.bin_failed_count[:, : self.max_bin_count].clone()
        probs_all = torch.nn.functional.pad(
            probs_all.unsqueeze(1),
            (0, self.cfg.adaptive_kernel_size - 1),
            mode="replicate",
        )
        probs_all = torch.nn.functional.conv1d(
            probs_all, self.kernel.view(1, 1, -1)
        ).squeeze(1)
        valid_mask = (
            torch.arange(self.max_bin_count, device=self.device)
            < self.bin_count_per_motion.unsqueeze(1)
        )
        probs_all = probs_all * valid_mask.float()
        probs_all = (
            probs_all
            / (probs_all.sum(dim=1, keepdim=True) + 1e-8)
            * (1 - self.cfg.adaptive_uniform_ratio)
        )
        probs_all = probs_all + (
            self.cfg.adaptive_uniform_ratio / self.bin_count_per_motion
        ).unsqueeze(1) * valid_mask.float()

        # Sample and assign per motion (loop over motions, but probs precomputed)
        for motion_idx in range(self.motion.num_motions):
            mask = bound_motion_ids == motion_idx
            if not mask.any():
                continue
            n_m = int(mask.sum())
            probs = probs_all[motion_idx, : self.bin_count_per_motion[motion_idx]]
            bins = torch.multinomial(probs, n_m, replacement=True)
            jitter = sample_uniform(0.0, 1.0, (n_m,), device=self.device)
            frac = (bins.float() + jitter) / self.bin_count_per_motion[motion_idx]
            motion_len = self.motion_lengths[motion_idx]
            time_in_motion = (frac * motion_len.float()).long().clamp(max=motion_len - 1)
            self.time_steps[env_ids[mask]] = (
                self.motion.motion_start_idx[motion_idx] + time_in_motion
            )

    def _update_command(self):
        """Update command and per-motion bin_failed_count."""
        self.time_steps += 1
        env_ids = torch.where(
            self.time_steps >= self.motion.motion_end_idx[self.motion_ids]
        )[0]
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
            1, len(self.cfg.body_names), 1
        )

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
        delta_ori_w = yaw_quat(
            quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
        )

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(
            delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
        )

        # Per-motion bin EMA update
        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed
            + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()


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

@configclass
class MultiMotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MultiMotionCommand

    asset_name: str = MISSING

    motion_files: list[str] = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    look_ahead_frames = 5
    look_ahead_frame_skips = 5

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    force_update_frequency: int = 100
    max_force: float = 20.0

    force_push_body: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    force_push_body_offset: list[list[float]] = [[0.18, -0.025, 0.0],[0.18, +0.025, 0.0], [0.0, 0.0, 0.35]]

    # vr 3 point related
    vr_3point_body: list[str] = ["left_wrist_yaw_link", "right_wrist_yaw_link", "torso_link"]
    vr_3point_body_offset: list[list[float]] = [[0.18, -0.025, 0.0],[0.18, +0.025, 0.0], [0.0, 0.0, 0.35]]

    # contact label related stuff
    has_wrist_grasp_label: bool = False


@configclass
class BindedMultiMotionCommandCfg(MultiMotionCommandCfg):
    """Configuration for the binded multi-motion command.

    Same as MultiMotionCommandCfg but each environment is permanently bound to a motion
    via env_id % num_motions. Use this when you want deterministic env-to-motion
    assignment (e.g., more envs than motions, wrap around).
    """

    class_type: type = BindedMultiMotionCommand