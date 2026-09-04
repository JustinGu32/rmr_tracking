"""Replay a G1 motion CSV through Isaac FK and save a local full-body NPZ.

CSV columns are pelvis position, pelvis quaternion in XYZW order, and the 29
joint positions in ``CSV_JOINT_NAMES`` order.  Unlike the original W&B
converter, this bounded variant writes directly to ``--output-path`` and saves
the resolved articulation joint/body orders as sidecars.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

CSV_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input-file", type=Path, required=True)
parser.add_argument("--input-sha256", required=True)
parser.add_argument("--input-fps", type=int, default=30)
parser.add_argument("--output-fps", type=int, default=50)
parser.add_argument("--output-path", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.input_file = args_cli.input_file.resolve()
args_cli.output_path = args_cli.output_path.resolve()
if not args_cli.input_file.is_file():
    parser.error(f"input file does not exist: {args_cli.input_file}")
if sha256_file(args_cli.input_file) != args_cli.input_sha256:
    parser.error("input file SHA-256 does not match --input-sha256")
for candidate in (
    args_cli.output_path,
    Path(str(args_cli.output_path) + ".joint_names.npy"),
    Path(str(args_cli.output_path) + ".body_names.npy"),
):
    if candidate.exists():
        parser.error(f"refusing to overwrite output: {candidate}")
if args_cli.input_fps <= 0 or args_cli.output_fps <= 0:
    parser.error("input and output fps must be positive")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg()
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0)
    )
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


class MotionLoader:
    def __init__(
        self,
        motion_file: Path,
        input_fps: int,
        output_fps: int,
        device: torch.device,
    ) -> None:
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        motion = torch.from_numpy(np.loadtxt(motion_file, delimiter=","))
        if motion.ndim != 2 or motion.shape[1] != 36 or motion.shape[0] < 2:
            raise ValueError("motion CSV must have shape (T >= 2, 36)")
        if not torch.isfinite(motion).all():
            raise ValueError("motion CSV must be finite")
        motion = motion.to(torch.float32).to(device)
        self.base_pos_input = motion[:, :3]
        self.base_quat_input = motion[:, 3:7][:, [3, 0, 1, 2]]
        self.joint_pos_input = motion[:, 7:36]
        self.input_frames = int(motion.shape[0])
        self.duration = (self.input_frames - 1) * self.input_dt
        self._interpolate(device)
        self._compute_velocities()

    def _interpolate(self, device: torch.device) -> None:
        times = torch.arange(
            0, self.duration, self.output_dt, device=device, dtype=torch.float32
        )
        self.output_frames = int(times.shape[0])
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(
            index_0 + 1,
            torch.tensor(self.input_frames - 1, device=device),
        )
        blend = phase * (self.input_frames - 1) - index_0
        self.base_pos = self._lerp(
            self.base_pos_input[index_0],
            self.base_pos_input[index_1],
            blend.unsqueeze(1),
        )
        self.base_quat = self._slerp(
            self.base_quat_input[index_0],
            self.base_quat_input[index_1],
            blend,
        )
        self.joint_pos = self._lerp(
            self.joint_pos_input[index_0],
            self.joint_pos_input[index_1],
            blend.unsqueeze(1),
        )

    @staticmethod
    def _lerp(a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        return a * (1.0 - blend) + b * blend

    @staticmethod
    def _slerp(
        a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
    ) -> torch.Tensor:
        result = torch.zeros_like(a)
        for index in range(a.shape[0]):
            result[index] = quat_slerp(a[index], b[index], blend[index])
        return result

    def _compute_velocities(self) -> None:
        self.base_lin_vel = torch.gradient(
            self.base_pos, spacing=self.output_dt, dim=0
        )[0]
        self.joint_vel = torch.gradient(
            self.joint_pos, spacing=self.output_dt, dim=0
        )[0]
        q_rel = quat_mul(self.base_quat[2:], quat_conjugate(self.base_quat[:-2]))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.base_ang_vel = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def next_state(self) -> tuple[tuple[torch.Tensor, ...], bool]:
        index = self.current_idx
        state = (
            self.base_pos[index : index + 1],
            self.base_quat[index : index + 1],
            self.base_lin_vel[index : index + 1],
            self.base_ang_vel[index : index + 1],
            self.joint_pos[index : index + 1],
            self.joint_vel[index : index + 1],
        )
        self.current_idx += 1
        finished = self.current_idx >= self.output_frames
        return state, finished


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    motion = MotionLoader(
        args_cli.input_file,
        args_cli.input_fps,
        args_cli.output_fps,
        sim.device,
    )
    robot = scene["robot"]
    joint_indices = robot.find_joints(CSV_JOINT_NAMES, preserve_order=True)[0]
    if len(joint_indices) != len(CSV_JOINT_NAMES):
        raise ValueError("Isaac articulation does not contain all CSV joints")
    log: dict[str, list[np.ndarray] | np.ndarray] = {
        "fps": np.asarray([args_cli.output_fps], dtype=np.int64),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    while simulation_app.is_running():
        (base_pos, base_quat, base_lin, base_ang, joint_pos_in, joint_vel_in), finished = (
            motion.next_state()
        )
        root_state = robot.data.default_root_state.clone()
        root_state[:, :3] = base_pos
        root_state[:, :2] += scene.env_origins[:, :2]
        root_state[:, 3:7] = base_quat
        root_state[:, 7:10] = base_lin
        root_state[:, 10:] = base_ang
        robot.write_root_state_to_sim(root_state)

        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, joint_indices] = joint_pos_in
        joint_vel[:, joint_indices] = joint_vel_in
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()
        scene.update(sim.get_physics_dt())

        for name in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            value = getattr(robot.data, name)[0].cpu().numpy().copy()
            assert isinstance(log[name], list)
            log[name].append(value)
        if finished:
            break

    arrays = {
        name: np.stack(value, axis=0) if isinstance(value, list) else value
        for name, value in log.items()
    }
    if arrays["joint_pos"].shape[0] != motion.output_frames:
        raise ValueError("logger did not capture every interpolated frame")
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Isaac FK output contains non-finite values")

    args_cli.output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = args_cli.output_path.with_name(args_cli.output_path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    temporary.replace(args_cli.output_path)
    np.save(str(args_cli.output_path) + ".joint_names.npy", np.asarray(robot.joint_names))
    np.save(str(args_cli.output_path) + ".body_names.npy", np.asarray(robot.body_names))
    print(
        f"[INFO] saved {args_cli.output_path} with "
        f"{motion.output_frames} frames, {len(robot.joint_names)} joints, "
        f"and {len(robot.body_names)} bodies"
    )
    print(f"[INFO] joint_names={robot.joint_names}")
    print(f"[INFO] body_names={robot.body_names}")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
