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

        # Staircase object placement (static): world pose = env_origin + box_position.
        self.box_position = torch.tensor(self.cfg.box_position, device=self.device)
        self.box_rotation = torch.tensor(self.cfg.box_rotation, dtype=torch.float32, device=self.device).repeat(
            self.num_envs, 1
        )

        # ── Stair-phase termination support ──────────────────────────────────────
        # Precompute, for every motion frame, which stair each foot SHOULD be on
        # ("on the stair, anywhere", gated by stance = low ref foot speed). Used by the
        # mdp.bad_stair_phase termination. Only built when stair_bounds are provided.
        self.stair_expected = None  # (T, 2) long: expected stair (1-based) per [L, R] foot, 0 = none
        self.foot_off_streak = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        if self.cfg.stair_bounds:
            self._build_stair_schedule()

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

    # --- Staircase object pose (static, per-env). Consumed by StaircaseEnv. ---
    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._env.scene.env_origins + self.box_position

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self.box_rotation

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return torch.zeros_like(self.object_pos_w)

    @property
    def object_ang_vel_w(self) -> torch.Tensor:
        return torch.zeros_like(self.object_pos_w)

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

    # ── Stair-phase schedule / membership (for bad_stair_phase termination) ──────
    def _stair_membership(self, pos_local: torch.Tensor) -> torch.Tensor:
        """Which stair (1-based) each point sits on; 0 = none. pos_local: (..., 3) in stair frame.

        "On the stair, anywhere": xy within the stair AABB (+ xy slack), z near the stair top.
        """
        out = torch.zeros(pos_local.shape[:-1], dtype=torch.long, device=self.device)
        xs, ys = self.cfg.stair_xy_slack, self.cfg.stair_xy_slack
        zs = self.cfg.stair_z_slack
        for i, (lo, hi) in enumerate(self.cfg.stair_bounds):
            lo = torch.tensor(lo, device=self.device)
            hi = torch.tensor(hi, device=self.device)
            on = (
                (pos_local[..., 0] >= lo[0] - xs) & (pos_local[..., 0] <= hi[0] + xs)
                & (pos_local[..., 1] >= lo[1] - ys) & (pos_local[..., 1] <= hi[1] + ys)
                & ((pos_local[..., 2] - hi[2]).abs() < zs)
            )
            out = torch.where(on, torch.full_like(out, i + 1), out)
        return out

    def _world_to_stair_local(self, pos_world: torch.Tensor) -> torch.Tensor:
        """Rotate world points into the staircase local frame (about box yaw, minus box position)."""
        w, z = self.cfg.box_rotation[0], self.cfg.box_rotation[3]
        yaw = 2.0 * math.atan2(z, w)
        c, s = math.cos(-yaw), math.sin(-yaw)
        d = pos_world - self.box_position
        lx = c * d[..., 0] - s * d[..., 1]
        ly = s * d[..., 0] + c * d[..., 1]
        return torch.stack([lx, ly, d[..., 2]], dim=-1)

    def _build_stair_schedule(self):
        """Precompute (T, 2) expected-stair-per-foot from the motion (stance frames only)."""
        L = self.cfg.body_names.index(self.cfg.stair_foot_body_names[0])
        R = self.cfg.body_names.index(self.cfg.stair_foot_body_names[1])
        L_abs = int(self.body_indexes[L].item())
        R_abs = int(self.body_indexes[R].item())
        pos = self.motion._body_pos_w  # (T, num_bodies, 3) in motion (== env-origin) frame
        vel = self.motion._body_lin_vel_w
        T = pos.shape[0]
        sched = torch.zeros(T, 2, dtype=torch.long, device=self.device)
        for col, abi in enumerate((L_abs, R_abs)):
            local = self._world_to_stair_local(pos[:, abi, :])         # (T, 3)
            stair = self._stair_membership(local)                       # (T,)
            speed = torch.linalg.norm(vel[:, abi, :], dim=-1)           # (T,)
            stance = speed < self.cfg.stair_foot_speed_thr
            sched[:, col] = torch.where(stance, stair, torch.zeros_like(stair))
        self.stair_expected = sched

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
        #self._adaptive_sampling(env_ids)

        self._uniform_sampling(env_ids)

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
        joint_vel += sample_uniform(*self.cfg.joint_velocity_range, joint_vel.shape, joint_vel.device)
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

        if self.stair_expected is not None:
            self._update_foot_off_streak(env_ids)

    def _update_foot_off_streak(self, resampled_env_ids: Sequence[int]):
        """Per step, track how long each foot has been OFF the stair the reference expects.

        For each foot: if the reference (at this env's time_step) expects it on stair S>0 but the
        robot foot is not on stair S, increment the off-streak; otherwise reset it. Resampled envs
        start fresh. Consumed by mdp.bad_stair_phase.
        """
        L = self.cfg.body_names.index(self.cfg.stair_foot_body_names[0])
        R = self.cfg.body_names.index(self.cfg.stair_foot_body_names[1])
        # expected stair per foot for each env at its current frame: (num_envs, 2)
        expected = self.stair_expected[self.time_steps]  # (num_envs, 2)
        # robot foot world pos -> stair-local -> membership
        foot_w = self.robot_body_pos_w[:, [L, R], :] - self._env.scene.env_origins[:, None, :]
        foot_local = self._world_to_stair_local(foot_w)             # (num_envs, 2, 3)
        robot_stair = self._stair_membership(foot_local)            # (num_envs, 2)
        violating = (expected > 0) & (robot_stair != expected)      # (num_envs, 2)
        self.foot_off_streak = torch.where(
            violating, self.foot_off_streak + 1, torch.zeros_like(self.foot_off_streak)
        )
        if len(resampled_env_ids) > 0:
            self.foot_off_streak[resampled_env_ids] = 0

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
    joint_velocity_range: tuple[float, float] = (0.0, 0.0)

    # Staircase object placement (world pose = env_origin + box_position).
    box_position: list[float] = [0.0, 0.0, 0.0]
    box_rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # ── Stair-phase termination schedule (empty stair_bounds -> feature disabled) ──
    # Each entry: ([xmin, ymin, zmin], [xmax, ymax, zmax]) in the staircase local frame.
    stair_bounds: list = []
    stair_foot_body_names: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link")
    stair_foot_speed_thr: float = 0.15  # ref foot speed below this = stance (should be planted)
    stair_xy_slack: float = 0.05        # "anywhere on the stair" xy tolerance
    stair_z_slack: float = 0.08         # vertical tolerance to the stair top

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    min_sample_idx: int = 0
    max_sample_idx: int = 0
    steps_collect: int  = 0

    # TODO: add config term for sampling (adaptive vs uniform)