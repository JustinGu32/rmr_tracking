"""Batch convert BONES-SEED G1 CSV files to a single Zarr motion store.

Uses MotionLoader (from csv_to_npz.py) for proven interpolation + velocity
computation, and MuJoCo FK for body positions/quaternions (verified identical
to Isaac Sim FK within 0.04°).

No GPU or Isaac Sim required — runs on CPU only.

Usage:
    python scripts/batch_csv_to_zarr.py \
        --csv_dir /move/data/bones/g1/csv/221116 \
        --output_path /tmp/test_motions.zarr \
        --max_clips 10 \
        --input_fps 120 \
        --output_fps 50
"""

import argparse
import os
import glob
import tempfile
import numpy as np
import torch
import zarr

from pathlib import Path
from scipy.spatial.transform import Rotation
from tqdm import tqdm

try:
    import mujoco
except ImportError:
    raise ImportError("MuJoCo is required. Install with: pip install mujoco")


# ─── Pure torch quaternion math (no Isaac Sim dependency) ─────────────────────

def quat_conjugate(q):
    """Conjugate of quaternion (wxyz format)."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)

def quat_mul(q1, q2):
    """Hamilton product of two quaternions (wxyz format)."""
    w1, x1, y1, z1 = q1[..., 0:1], q1[..., 1:2], q1[..., 2:3], q1[..., 3:4]
    w2, x2, y2, z2 = q2[..., 0:1], q2[..., 1:2], q2[..., 2:3], q2[..., 3:4]
    return torch.cat([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dim=-1)

def axis_angle_from_quat(q):
    """Convert quaternion (wxyz) to axis-angle."""
    # Ensure w >= 0 for numerical stability
    q = torch.where(q[..., :1] < 0, -q, q)
    w = q[..., :1].clamp(-1, 1)
    xyz = q[..., 1:]
    sin_half = xyz.norm(dim=-1, keepdim=True).clamp(min=1e-10)
    angle = 2.0 * torch.atan2(sin_half, w)
    axis = xyz / sin_half
    return axis * angle

def quat_slerp(q0, q1, t):
    """Slerp between two quaternions (wxyz format)."""
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(0, 1)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp(min=1e-10)
    s0 = torch.sin((1 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    # Fallback to lerp for small angles
    close = (dot > 0.9999)
    s0 = torch.where(close, 1.0 - t, s0)
    s1 = torch.where(close, t, s1)
    result = s0 * q0 + s1 * q1
    return result / result.norm(dim=-1, keepdim=True)

G1_XML_PATH = "/move/u/takaraet/kimodo/kimodo/assets/skeletons/g1skel34/xml/g1.xml"


# ─── MotionLoader (from csv_to_npz.py) ───────────────────────────────────────

class MotionLoader:
    """Loads, interpolates, and computes velocities for a motion CSV.
    Identical to csv_to_npz.py's MotionLoader class.
    Expects CSV: no header, columns = root_pos(3), root_quat_xyzw(4), joints(29)
    """
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
        if self.has_object:
            self.motion_object_poss_input = motion[:, 36:39]
            self.motion_object_rots_input = motion[:, 39:43][:, [3, 0, 1, 2]]
        self.input_frames = motion.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate_motion(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0], self.motion_base_poss_input[index_1], blend.unsqueeze(1))
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0], self.motion_base_rots_input[index_1], blend)
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0], self.motion_dof_poss_input[index_1], blend.unsqueeze(1))

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        slerped = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped

    def _compute_frame_blend(self, times):
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations, dt):
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        return torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_all_states(self):
        """Return all interpolated states as numpy arrays for batch FK."""
        return {
            "base_pos": self.motion_base_poss.cpu().numpy(),        # (T, 3)
            "base_rot": self.motion_base_rots.cpu().numpy(),        # (T, 4) wxyz
            "dof_pos": self.motion_dof_poss.cpu().numpy(),          # (T, 29)
            "dof_vel": self.motion_dof_vels.cpu().numpy(),          # (T, 29)
        }


# ─── BONES CSV → MotionLoader format ─────────────────────────────────────────

def convert_bones_csv_to_standard(csv_path: str, tmp_dir: str) -> str:
    """Convert a BONES CSV (header, cm, degrees, euler) to MotionLoader format.

    MotionLoader expects: no header, columns = root_pos(3), root_quat_xyzw(4), joints_rad(29)
    BONES has: header row, Frame(1), root_pos_cm(3), root_euler_deg(3), joints_deg(29)
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    root_pos_m = data[:, 1:4] / 100.0
    root_euler_deg = data[:, 4:7]
    joint_rad = np.deg2rad(data[:, 7:36])

    # BONES CSV uses intrinsic ZYX Euler convention
    # (see seed-viewer/frontend/public/src/models.ts line 580)
    root_rot = Rotation.from_euler("ZYX", root_euler_deg[:, [2, 1, 0]], degrees=True)
    root_quat_xyzw = root_rot.as_quat()

    T = data.shape[0]
    standard = np.zeros((T, 36), dtype=np.float64)
    standard[:, 0:3] = root_pos_m
    standard[:, 3:7] = root_quat_xyzw
    standard[:, 7:36] = joint_rad

    basename = Path(csv_path).stem
    tmp_path = os.path.join(tmp_dir, f"{basename}_std.csv")
    np.savetxt(tmp_path, standard, delimiter=",")
    return tmp_path


# ─── MuJoCo FK ───────────────────────────────────────────────────────────────

def run_mujoco_fk_batch(model, data_mj, states: dict) -> dict:
    """Run MuJoCo FK for all frames. Returns body pos/quat arrays."""
    T = states["base_pos"].shape[0]
    n_bodies = model.nbody
    body_pos = np.zeros((T, n_bodies, 3), dtype=np.float32)
    body_quat = np.zeros((T, n_bodies, 4), dtype=np.float32)

    for t in range(T):
        qpos = np.zeros(36, dtype=np.float64)
        qpos[0:3] = states["base_pos"][t]
        qpos[3:7] = states["base_rot"][t]   # wxyz (MuJoCo format)
        qpos[7:36] = states["dof_pos"][t]
        data_mj.qpos[:] = qpos
        mujoco.mj_forward(model, data_mj)
        body_pos[t] = data_mj.xpos.copy()
        body_quat[t] = data_mj.xquat.copy()

    return {"body_pos_w": body_pos, "body_quat_w": body_quat}


def compute_body_velocities(body_pos, body_quat, dt):
    """Compute body velocities via gradient + SO3 derivative (numpy)."""
    body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)

    # Angular velocity via SO3 derivative (same method as MotionLoader)
    T, N = body_quat.shape[:2]
    body_ang_vel = np.zeros((T, N, 3), dtype=np.float32)
    for b in range(N):
        q = body_quat[:, b]  # (T, 4) wxyz
        # Convert to torch for axis_angle_from_quat
        q_t = torch.from_numpy(q).float()
        q_prev, q_next = q_t[:-2], q_t[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
        body_ang_vel[:, b] = omega.numpy()

    return body_lin_vel, body_ang_vel


# ─── Process single clip ─────────────────────────────────────────────────────

def process_clip(csv_path, model, data_mj, input_fps, output_fps, tmp_dir, device,
                 body_reorder_idx=None, joint_reorder_idx=None):
    """Convert BONES CSV → MotionLoader → MuJoCo FK → motion dict."""
    # Convert BONES CSV to standard format
    std_csv = convert_bones_csv_to_standard(csv_path, tmp_dir)

    # MotionLoader: interpolation + velocity computation
    motion = MotionLoader(std_csv, input_fps, output_fps, device, frame_range=None)
    states = motion.get_all_states()

    # MuJoCo FK
    fk = run_mujoco_fk_batch(model, data_mj, states)

    # Reorder bodies to Isaac ordering
    if body_reorder_idx is not None:
        fk["body_pos_w"] = fk["body_pos_w"][:, body_reorder_idx]
        fk["body_quat_w"] = fk["body_quat_w"][:, body_reorder_idx]

    # Reorder joints to Isaac ordering
    joint_pos = states["dof_pos"].astype(np.float32)
    joint_vel = states["dof_vel"].astype(np.float32)
    if joint_reorder_idx is not None:
        joint_pos = joint_pos[:, joint_reorder_idx]
        joint_vel = joint_vel[:, joint_reorder_idx]

    # Body velocities
    dt = 1.0 / output_fps
    body_lin_vel, body_ang_vel = compute_body_velocities(
        fk["body_pos_w"], fk["body_quat_w"], dt)

    # Clean up temp file
    os.remove(std_csv)

    return {
        "body_pos_w": fk["body_pos_w"],
        "body_quat_w": fk["body_quat_w"],
        "body_lin_vel_w": body_lin_vel,
        "body_ang_vel_w": body_ang_vel,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "fps": output_fps,
    }


# ─── Streaming Zarr writer (avoids OOM) ───────────────────────────────────────

ARRAY_KEYS = ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
              "body_lin_vel_w", "body_ang_vel_w"]


def init_zarr(output_path, output_fps, body_names):
    """Create empty Zarr store with resizable arrays."""
    store = zarr.DirectoryStore(output_path)
    root = zarr.group(store=store, overwrite=True)
    root.create_dataset("fps", data=np.array([output_fps], dtype=np.int32))
    root.create_dataset("body_names", data=np.array(body_names, dtype=object),
                        object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, 'VLenUTF8') else None)
    return root


def append_batch_to_zarr(root, batch_motions, batch_names, batch_meta):
    """Append a batch of clips to the Zarr store. Creates arrays on first call."""
    if not batch_motions:
        return

    # Concatenate batch
    batch_arrays = {}
    for key in ARRAY_KEYS:
        batch_arrays[key] = np.concatenate([m[key] for m in batch_motions])

    # Compute clip start/end indices for this batch
    batch_starts, batch_ends = [], []
    # Get current total frames from existing data (0 if first batch)
    if "joint_pos" in root:
        offset = root["joint_pos"].shape[0]
    else:
        offset = 0

    cursor = offset
    for m in batch_motions:
        batch_starts.append(cursor)
        cursor += m["body_pos_w"].shape[0]
        batch_ends.append(cursor)

    compressor = zarr.Blosc(cname="zstd", clevel=3)
    chunk_frames = 10000

    for key in ARRAY_KEYS:
        arr = batch_arrays[key]
        if key not in root:
            # Create resizable array
            chunks = (chunk_frames,) + arr.shape[1:]
            root.create_dataset(key, data=arr, chunks=chunks, dtype="float32",
                               compressor=compressor, maxshape=(None,) + arr.shape[1:])
        else:
            root[key].append(arr, axis=0)

    # Append clip metadata
    for meta_key in ["clip_start_idx", "clip_end_idx"]:
        data = np.array(batch_starts if "start" in meta_key else batch_ends, dtype=np.int64)
        if meta_key not in root:
            root.create_dataset(meta_key, data=data, chunks=(chunk_frames,),
                               maxshape=(None,), dtype="int64")
        else:
            root[meta_key].append(data, axis=0)

    # Append clip names
    names_arr = np.array(batch_names, dtype=object)
    if "clip_names" not in root:
        root.create_dataset("clip_names", data=names_arr,
                           object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, 'VLenUTF8') else None,
                           maxshape=(None,))
    else:
        root["clip_names"].append(names_arr, axis=0)

    # Append per-clip metadata (int arrays like content_props)
    for meta_key, values in batch_meta.items():
        if meta_key.endswith("_desc"):
            # String metadata
            data = np.array(values, dtype=object)
            if meta_key not in root:
                root.create_dataset(meta_key, data=data,
                                   object_codec=zarr.codecs.VLenUTF8() if hasattr(zarr.codecs, 'VLenUTF8') else None,
                                   maxshape=(None,))
            else:
                root[meta_key].append(data, axis=0)
        else:
            # Int metadata
            data = np.array(values, dtype=np.int32)
            if meta_key not in root:
                root.create_dataset(meta_key, data=data, chunks=(chunk_frames,),
                                   maxshape=(None,), dtype="int32")
            else:
                root[meta_key].append(data, axis=0)


# ─── Parallel worker ──────────────────────────────────────────────────────────

_worker_model = None
_worker_data = None
_worker_tmp_dir = None
_worker_body_reorder_idx = None
_worker_joint_reorder_idx = None


def _worker_init(g1_xml, body_reorder_idx=None, joint_reorder_idx=None):
    """Initialize per-worker MuJoCo model (can't share across processes)."""
    global _worker_model, _worker_data, _worker_tmp_dir, _worker_body_reorder_idx, _worker_joint_reorder_idx
    _worker_model = mujoco.MjModel.from_xml_path(g1_xml)
    _worker_data = mujoco.MjData(_worker_model)
    _worker_tmp_dir = tempfile.mkdtemp()
    _worker_body_reorder_idx = body_reorder_idx
    _worker_joint_reorder_idx = joint_reorder_idx


def _worker_process(args):
    """Process a single clip in a worker process."""
    csv_path, input_fps, output_fps = args
    try:
        clip = process_clip(csv_path, _worker_model, _worker_data,
                           input_fps, output_fps, _worker_tmp_dir, torch.device("cpu"),
                           body_reorder_idx=_worker_body_reorder_idx,
                           joint_reorder_idx=_worker_joint_reorder_idx)
        return (Path(csv_path).stem, clip, None)
    except Exception as e:
        return (Path(csv_path).stem, None, str(e))


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description="Batch convert BONES CSVs to Zarr (MotionLoader + MuJoCo FK)")
    parser.add_argument("--csv_dir", type=str, required=True,
                        help="Directory containing CSVs (searched recursively)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Output Zarr store path")
    parser.add_argument("--input_fps", type=int, default=120)
    parser.add_argument("--output_fps", type=int, default=50)
    parser.add_argument("--max_clips", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of parallel workers (default: 16)")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Clips per batch before flushing to Zarr (default: 5000)")
    parser.add_argument("--metadata_csv", type=str,
                        default="/move/data/bones/metadata/seed_metadata_v003.csv",
                        help="BONES metadata CSV for per-clip attributes")
    parser.add_argument("--g1_xml", type=str, default=G1_XML_PATH)
    parser.add_argument("--body_names", type=str, nargs="+", default=[
        # Isaac body ordering (from robot.body_names at runtime)
        "pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "waist_yaw_link",
        "left_hip_roll_link", "right_hip_roll_link", "waist_roll_link",
        "left_hip_yaw_link", "right_hip_yaw_link", "torso_link",
        "left_knee_link", "right_knee_link",
        "left_shoulder_pitch_link", "right_shoulder_pitch_link",
        "left_ankle_pitch_link", "right_ankle_pitch_link",
        "left_shoulder_roll_link", "right_shoulder_roll_link",
        "left_ankle_roll_link", "right_ankle_roll_link",
        "left_shoulder_yaw_link", "right_shoulder_yaw_link",
        "left_elbow_link", "right_elbow_link",
        "left_wrist_roll_link", "right_wrist_roll_link",
        "left_wrist_pitch_link", "right_wrist_pitch_link",
        "left_wrist_yaw_link", "right_wrist_yaw_link",
    ], help="Body names to include in Zarr (in this order). Default: Isaac body ordering.")
    parser.add_argument("--joint_names", type=str, nargs="+", default=[
        # Isaac joint ordering (from robot.joint_names at runtime)
        "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
        "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
        "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
        "left_knee_joint", "right_knee_joint",
        "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
        "left_ankle_pitch_joint", "right_ankle_pitch_joint",
        "left_shoulder_roll_joint", "right_shoulder_roll_joint",
        "left_ankle_roll_joint", "right_ankle_roll_joint",
        "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
        "left_elbow_joint", "right_elbow_joint",
        "left_wrist_roll_joint", "right_wrist_roll_joint",
        "left_wrist_pitch_joint", "right_wrist_pitch_joint",
        "left_wrist_yaw_joint", "right_wrist_yaw_joint",
    ], help="Joint names in Isaac ordering for reordering.")
    parser.add_argument("--categories", type=str, nargs="+", default=None,
                        help="Only include clips whose metadata 'category' matches one of these (case-insensitive substring). "
                             "Example: --categories 'Basic Locomotion' 'Advanced Locomotion'")
    args = parser.parse_args()

    # Load metadata
    meta_lookup = {}
    if os.path.exists(args.metadata_csv):
        import pandas as pd
        meta_df = pd.read_csv(args.metadata_csv)
        for _, row in meta_df.iterrows():
            if pd.notna(row.get("move_g1_path")):
                stem = os.path.splitext(os.path.basename(row["move_g1_path"]))[0]
                meta_lookup[stem] = row.to_dict()
        print(f"Loaded metadata: {len(meta_lookup)} entries")

    csv_files = sorted(glob.glob(os.path.join(args.csv_dir, "**", "*.csv"), recursive=True))

    # Filter by metadata category if requested
    if args.categories and meta_lookup:
        cats_lower = [c.lower() for c in args.categories]
        filtered = []
        for f in csv_files:
            stem = os.path.splitext(os.path.basename(f))[0]
            if stem in meta_lookup:
                clip_cat = str(meta_lookup[stem].get("category", "")).lower()
                if any(c in clip_cat for c in cats_lower):
                    filtered.append(f)
        print(f"Category filter {args.categories}: {len(filtered)}/{len(csv_files)} clips match")
        csv_files = filtered

    if args.max_clips:
        csv_files = csv_files[:args.max_clips]
    print(f"Found {len(csv_files)} CSV files")
    if not csv_files:
        return

    assert os.path.exists(args.g1_xml), f"G1 XML not found: {args.g1_xml}"
    model = mujoco.MjModel.from_xml_path(args.g1_xml)
    mj_body_names = [model.body(i).name for i in range(model.nbody)]
    print(f"MuJoCo: {model.nbody} bodies, {model.njnt} joints, nq={model.nq}")

    # Body filter/reorder: if --body_names given, reorder FK output; else keep all
    if args.body_names is not None:
        body_reorder_idx = [mj_body_names.index(name) for name in args.body_names]
        zarr_body_names = args.body_names
        print(f"Body reorder: {dict(zip(args.body_names, body_reorder_idx))}")
    else:
        body_reorder_idx = None
        zarr_body_names = mj_body_names

    # Init Zarr store
    root = init_zarr(args.output_path, args.output_fps, zarr_body_names)

    # Joint reorder: MotionLoader/BONES order → Isaac order
    BONES_JOINT_NAMES = [
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
    joint_reorder_idx = [BONES_JOINT_NAMES.index(name) for name in args.joint_names]
    print(f"Joint reorder: {dict(zip(args.joint_names, joint_reorder_idx))}")

    num_workers = min(args.num_workers, len(csv_files))
    work_args = [(csv, args.input_fps, args.output_fps) for csv in csv_files]
    total_clips = 0
    total_frames = 0
    errors = 0

    # Batch buffer
    batch_motions = []
    batch_names = []

    def flush_batch():
        nonlocal total_clips, total_frames
        if not batch_motions:
            return
        # Build per-clip metadata for this batch
        batch_meta = {}
        if meta_lookup:
            props = []
            props_desc = []
            for name in batch_names:
                if name in meta_lookup and "content_props" in meta_lookup[name]:
                    val = meta_lookup[name]["content_props"]
                    if isinstance(val, (int, float)) and val == 0:
                        props.append(0)
                        props_desc.append("")
                    elif isinstance(val, float) and val != val:  # NaN
                        props.append(-1)
                        props_desc.append("")
                    else:
                        props.append(1)
                        props_desc.append(str(val))
                else:
                    props.append(-1)
                    props_desc.append("")
            batch_meta["content_props"] = props
            batch_meta["content_props_desc"] = props_desc

        frames_in_batch = sum(m["body_pos_w"].shape[0] for m in batch_motions)
        append_batch_to_zarr(root, batch_motions, batch_names, batch_meta)
        total_clips += len(batch_motions)
        total_frames += frames_in_batch
        print(f"\n  Flushed {len(batch_motions)} clips ({frames_in_batch} frames) → "
              f"total: {total_clips} clips, {total_frames} frames")
        batch_motions.clear()
        batch_names.clear()

    print(f"Using {num_workers} parallel workers, batch_size={args.batch_size}")
    with mp.Pool(num_workers, initializer=_worker_init, initargs=(args.g1_xml, body_reorder_idx, joint_reorder_idx)) as pool:
        for name, clip, err in tqdm(
            pool.imap_unordered(_worker_process, work_args),
            total=len(work_args), desc="Processing clips"
        ):
            if clip is not None:
                batch_motions.append(clip)
                batch_names.append(name)
                if len(batch_motions) >= args.batch_size:
                    flush_batch()
            else:
                errors += 1

    # Flush remaining
    flush_batch()

    print(f"\nDone! {total_clips} clips, {total_frames} frames, {errors} errors")
    print(f"Zarr: {args.output_path}")


if __name__ == "__main__":
    main()


