"""Compare MuJoCo FK vs Isaac Sim FK — both using MotionLoader.

Usage:
    python scripts/compare_fk.py \
        --csv_file /move/data/bones/g1/csv/221116/body_stretch_1_001__A062.csv \
        --input_fps 120 --output_fps 50 --headless
"""

import argparse
import os
import tempfile
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare MuJoCo FK vs Isaac Sim FK using MotionLoader")
parser.add_argument("--csv_file", type=str, required=True)
parser.add_argument("--input_fps", type=int, default=120)
parser.add_argument("--output_fps", type=int, default=50)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from pathlib import Path
from scipy.spatial.transform import Rotation

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]


# ─── BONES CSV conversion (ZYX Euler) ────────────────────────────────────────

def convert_bones_csv(csv_path, tmp_dir):
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    root_pos_m = data[:, 1:4] / 100.0
    root_euler_deg = data[:, 4:7]
    joint_rad = np.deg2rad(data[:, 7:36])
    root_rot = Rotation.from_euler("ZYX", root_euler_deg[:, [2, 1, 0]], degrees=True)
    root_quat_xyzw = root_rot.as_quat()
    T = data.shape[0]
    out = np.zeros((T, 36), dtype=np.float64)
    out[:, 0:3] = root_pos_m
    out[:, 3:7] = root_quat_xyzw
    out[:, 7:36] = joint_rad
    tmp_path = os.path.join(tmp_dir, f"{Path(csv_path).stem}_std.csv")
    np.savetxt(tmp_path, out, delimiter=",")
    return tmp_path


# ─── MotionLoader (from csv_to_npz.py) ───────────────────────────────────────

class MotionLoader:
    def __init__(self, motion_file, input_fps, output_fps, device, frame_range):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        if self.frame_range is None:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
        else:
            motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=",",
                skiprows=self.frame_range[0]-1, max_rows=self.frame_range[1]-self.frame_range[0]+1))
        motion = motion.to(torch.float32).to(self.device)
        self.motion_base_poss_input = motion[:, :3]
        self.motion_base_rots_input = motion[:, 3:7][:, [3, 0, 1, 2]]  # xyzw → wxyz
        self.motion_dof_poss_input = motion[:, 7:36]
        self.has_object = motion.shape[1] > 36
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        self.motion_base_poss = self.motion_base_poss_input[index_0] * (1-blend.unsqueeze(1)) + self.motion_base_poss_input[index_1] * blend.unsqueeze(1)
        slerped = torch.zeros(self.output_frames, 4, device=self.device)
        for i in range(self.output_frames):
            slerped[i] = quat_slerp(self.motion_base_rots_input[index_0[i]], self.motion_base_rots_input[index_1[i]], blend[i])
        self.motion_base_rots = slerped
        self.motion_dof_poss = self.motion_dof_poss_input[index_0] * (1-blend.unsqueeze(1)) + self.motion_dof_poss_input[index_1] * blend.unsqueeze(1)

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        q_prev, q_next = self.motion_base_rots[:-2], self.motion_base_rots[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx:self.current_idx+1],
            self.motion_base_rots[self.current_idx:self.current_idx+1],
            self.motion_base_lin_vels[self.current_idx:self.current_idx+1],
            self.motion_base_ang_vels[self.current_idx:self.current_idx+1],
            self.motion_dof_poss[self.current_idx:self.current_idx+1],
            self.motion_dof_vels[self.current_idx:self.current_idx+1],
        )
        self.current_idx += 1
        reset = self.current_idx >= self.output_frames
        if reset:
            self.current_idx = 0
        return state, None, reset


# ─── MuJoCo FK ───────────────────────────────────────────────────────────────

def run_mujoco_fk(motion):
    import mujoco
    g1_xml = "/move/u/takaraet/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"
    model = mujoco.MjModel.from_xml_path(g1_xml)
    data = mujoco.MjData(model)
    n_bodies = model.nbody
    body_names = [model.body(i).name for i in range(n_bodies)]
    T = motion.output_frames
    body_pos = np.zeros((T, n_bodies, 3), dtype=np.float32)
    body_quat = np.zeros((T, n_bodies, 4), dtype=np.float32)
    motion.current_idx = 0
    for t in range(T):
        state, _, _ = motion.get_next_state()
        bp, br, _, _, dp, _ = state
        qpos = np.zeros(36, dtype=np.float64)
        qpos[0:3] = bp[0].cpu().numpy()
        qpos[3:7] = br[0].cpu().numpy()
        qpos[7:36] = dp[0].cpu().numpy()
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        body_pos[t] = data.xpos.copy()
        body_quat[t] = data.xquat.copy()
    return {"body_pos_w": body_pos, "body_quat_w": body_quat, "body_names": body_names, "T": T}


# ─── Isaac Sim FK ─────────────────────────────────────────────────────────────

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"))
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_isaac_fk(motion, sim, scene):
    robot = scene["robot"]
    ji = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
    body_names = list(robot.data.body_names)
    T = motion.output_frames
    nb = len(body_names)
    body_pos = np.zeros((T, nb, 3), dtype=np.float32)
    body_quat = np.zeros((T, nb, 4), dtype=np.float32)
    motion.current_idx = 0
    for t in range(T):
        state, _, _ = motion.get_next_state()
        bp, br, blv, bav, dp, dv = state
        rs = robot.data.default_root_state.clone()
        rs[0, :3] = bp[0]; rs[0, :2] += scene.env_origins[0, :2]
        rs[0, 3:7] = br[0]; rs[0, 7:10] = blv[0]; rs[0, 10:] = bav[0]
        robot.write_root_state_to_sim(rs)
        jp = robot.data.default_joint_pos.clone()
        jv = robot.data.default_joint_vel.clone()
        jp[0, ji] = dp[0]; jv[0, ji] = dv[0]
        robot.write_joint_state_to_sim(jp, jv)
        sim.render(); scene.update(sim.get_physics_dt())
        body_pos[t] = robot.data.body_pos_w[0].cpu().numpy()
        body_quat[t] = robot.data.body_quat_w[0].cpu().numpy()
    return {"body_pos_w": body_pos, "body_quat_w": body_quat, "body_names": body_names, "T": T}


# ─── Compare ─────────────────────────────────────────────────────────────────

def compare(mj, isaac):
    T = min(mj["T"], isaac["T"])
    common = [(n, i, mj["body_names"].index(n)) for i, n in enumerate(isaac["body_names"]) if n in mj["body_names"]]
    print(f"\n{'='*65}")
    print(f"MuJoCo vs Isaac Sim FK (both via MotionLoader, {T} frames)")
    print(f"{'='*65}")
    print(f"{'Body':<30} {'Pos L2 (m)':<15} {'Quat (deg)':<15}")
    print("-" * 60)
    pe_all, qe_all = [], []
    for name, ii, mi in common:
        pe = np.linalg.norm(isaac["body_pos_w"][:T, ii] - mj["body_pos_w"][:T, mi], axis=-1).mean()
        dot = np.clip(np.abs(np.sum(isaac["body_quat_w"][:T, ii] * mj["body_quat_w"][:T, mi], axis=-1)), 0, 1)
        qe = np.degrees(2 * np.arccos(dot)).mean()
        pe_all.append(pe); qe_all.append(qe)
        print(f"{name:<30} {pe:<15.4f} {qe:<15.2f}")
    print("-" * 60)
    print(f"{'MEAN':<30} {np.mean(pe_all):<15.4f} {np.mean(qe_all):<15.2f}")
    print(f"{'MAX':<30} {np.max(pe_all):<15.4f} {np.max(qe_all):<15.2f}")


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    with tempfile.TemporaryDirectory() as tmp:
        std = convert_bones_csv(args_cli.csv_file, tmp)
        motion = MotionLoader(std, args_cli.input_fps, args_cli.output_fps, sim.device, None)
        print(f"MotionLoader: {motion.output_frames} frames")
        print("\n--- MuJoCo FK ---")
        mj = run_mujoco_fk(motion)
        print("--- Isaac Sim FK ---")
        isaac = run_isaac_fk(motion, sim, scene)
    compare(mj, isaac)

if __name__ == "__main__":
    main()
    simulation_app.close()
