from __future__ import annotations

from pathlib import Path
import torch
import torch.nn.functional as F
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms, quat_apply, quat_inv, quat_mul

from whole_body_tracking.tasks.staircase.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    grav_dir = quat_apply(quat_inv(command.robot_anchor_quat_w), command.down_dir)
    return grav_dir.view(env.num_envs, -1)


def vr_3point_local_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:, None, :]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    return ref_3point_root.view(env.num_envs, -1)

def vr_3point_local_compliant_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ext_force_disp_w = command.last_force_applied * command.eef_stiffness_buf[:,:,None]
    root_quat = command.robot_anchor_quat_w[:,None,:].repeat(1, len(command.cfg.vr_3point_body),1)
    ext_force_disp_l = quat_apply(quat_inv(root_quat), ext_force_disp_w)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_diff = command.vr_3point_body_pos_w - command.anchor_pos_w[:,None,:]
    ref_3point_root = quat_apply(quat_inv(ref_root_quat), ref_3point_diff)
    ref_3point_root -= ext_force_disp_l
    return ref_3point_root.view(env.num_envs, -1)

def vr_3point_local_orn_target(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_root_quat = command.anchor_quat_w.view(env.num_envs, 1, 4).repeat(1, len(command.cfg.vr_3point_body), 1)
    ref_3point_quat = command.vr_3point_body_quat_w
    ref_3point_root = quat_mul(quat_inv(ref_root_quat),ref_3point_quat)
    return ref_3point_root.view(env.num_envs, -1)

def wrist_grasp_label(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.wrist_grasp_label.view(env.num_envs, -1)

def compliance(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.eef_stiffness_buf * 10.0


def _save_depth_debug_frame(depth_image: torch.Tensor, frame_path: Path) -> None:
    """Save a normalized single-channel depth image for quick visual inspection."""
    try:
        from PIL import Image
    except ImportError:
        return

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    image_uint8 = (depth_image.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()
    Image.fromarray(image_uint8, mode="L").save(frame_path)


def depth_observation(
    env: ManagerBasedEnv,
    sensor_cfg,
    data_type: str = "distance_to_camera",
    min_depth_m: float = 0.2,
    max_depth_m: float = 4.0,
    resized_height: int | None = None,
    resized_width: int | None = None,
    flatten: bool = True,
    invert: bool = True,
    save_debug_frames: bool = False,
    debug_frame_dir: str = "data/debug_depth_frames",
    debug_max_frames: int = 4,
) -> torch.Tensor:
    """Return a clipped, normalized, optional downsampled depth observation."""
    sensor = env.scene.sensors[sensor_cfg.name]
    depth = sensor.data.output[data_type].float()
    if depth.ndim == 3:
        depth = depth.unsqueeze(-1)

    invalid_mask = ~torch.isfinite(depth)
    invalid_mask |= depth <= 0.0
    depth = torch.nan_to_num(depth, nan=max_depth_m, posinf=max_depth_m, neginf=min_depth_m)
    depth = torch.clamp(depth, min=min_depth_m, max=max_depth_m)

    depth_nchw = depth.permute(0, 3, 1, 2)
    if resized_height is not None and resized_width is not None:
        if depth_nchw.shape[-2:] != (resized_height, resized_width):
            depth_nchw = F.interpolate(depth_nchw, size=(resized_height, resized_width), mode="area")

    depth_norm = (depth_nchw - min_depth_m) / max(max_depth_m - min_depth_m, 1.0e-6)
    depth_norm = depth_norm.clamp(0.0, 1.0)
    if invert:
        depth_norm = 1.0 - depth_norm

    depth_obs = depth_norm.permute(0, 2, 3, 1).contiguous()
    if flatten:
        depth_obs = depth_obs.view(env.num_envs, -1)

    nonfinite_count = int((~torch.isfinite(depth_obs)).sum().item())
    if nonfinite_count:
        print(f"[DEPTH_OBS_DEBUG] Non-finite values after preprocessing: {nonfinite_count}. Replacing with zeros.")
        depth_obs = torch.nan_to_num(depth_obs, nan=0.0, posinf=0.0, neginf=0.0)

    env._last_depth_obs_mean = float(depth_obs.mean().item())
    env._last_depth_obs_std = float(depth_obs.std().item())
    env._last_depth_obs_batch_std = float(depth_obs.mean(dim=1).std().item())
    env._last_depth_obs_nonfinite_count = nonfinite_count

    if not hasattr(env, "_depth_obs_debug_printed"):
        invalid_ratio = float(invalid_mask.float().mean().item())
        print(
            "[DEPTH_OBS_DEBUG] "
            f"raw_shape={tuple(depth.shape)} processed_shape={tuple(depth_obs.shape)} "
            f"invalid_ratio={invalid_ratio:.4f} mean={env._last_depth_obs_mean:.4f} "
            f"std={env._last_depth_obs_std:.4f} batch_mean_std={env._last_depth_obs_batch_std:.4f}"
        )
        env._depth_obs_debug_printed = True

    if save_debug_frames:
        saved_frames = getattr(env, "_depth_obs_saved_frames", 0)
        if saved_frames < debug_max_frames:
            debug_image = depth_norm[0, 0]
            frame_path = Path(debug_frame_dir) / f"depth_env0_frame_{saved_frames:03d}.png"
            _save_depth_debug_frame(debug_image, frame_path)
            print(f"[DEPTH_OBS_DEBUG] Saved {frame_path}")
            env._depth_obs_saved_frames = saved_frames + 1

    return depth_obs
