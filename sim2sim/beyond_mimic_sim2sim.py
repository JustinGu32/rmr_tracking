import argparse
import json
import time
import numpy as np
import mujoco
import torch
from collections import deque
import mujoco.viewer as mjv
from tqdm import tqdm
import os
import pathlib

import onnx
import onnxruntime as ort


# ── Standalone math utilities (from isaaclab.utils.math) ──

def normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return normalize(quat_conjugate(q))


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)
    return torch.stack([w, x, y, z], dim=-1).view(shape)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def subtract_frame_transforms(t01, q01, t02=None, q02=None):
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


def motion_anchor_pos_b(
    robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w
) -> torch.Tensor:
    pos, _ = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        anchor_pos_w,
        anchor_quat_w,
    )
    return pos.view(1, -1)


def motion_anchor_ori_b(
    robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w
) -> torch.Tensor:
    _, ori = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        anchor_pos_w,
        anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def base_lin_vel(root_link_quat_w, root_lin_vel_w) -> torch.Tensor:
    """Root linear velocity in the asset's root frame."""
    return quat_rotate_inverse(root_link_quat_w, root_lin_vel_w)


def base_ang_vel(root_link_quat_w, root_ang_vel_w) -> torch.Tensor:
    """Root angular velocity in the asset's root frame."""
    # extract the used quantities (to enable type-hinting)
    return quat_rotate_inverse(root_link_quat_w, root_ang_vel_w)


# ── Walk-combined ONNX motion boundaries ──
# 12 LAFAN1 walk clips concatenated at 50fps
WALK_COMBINED_MOTIONS = {
    "walk1_subject1": (0, 13064),
    "walk1_subject2": (13065, 26129),
    "walk1_subject5": (26130, 39194),
    "walk2_subject1": (39195, 51103),
    "walk2_subject3": (51104, 63012),
    "walk2_subject4": (63013, 74921),
    "walk3_subject1": (74922, 87251),
    "walk3_subject2": (87252, 99581),
    "walk3_subject3": (99582, 111911),
    "walk3_subject4": (111912, 124241),
    "walk3_subject5": (124242, 136571),
    "walk4_subject1": (136572, 144766),
}


def joint_pos_rel(joint_pos, default_joint_pos) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their positions returned.
    """
    return joint_pos - default_joint_pos


def joint_vel_rel(joint_vel):
    """The joint velocities of the asset w.r.t. the default joint velocities.

    Note: Only the joints configured in :attr:`asset_cfg.joint_ids` will have their velocities returned.
    """
    return joint_vel


# -------------------------------------------------------------------
# Main low-level policy controller that:
#   - reads mimic obs from Redis
#   - feeds into policy
#   - runs the sim
# -------------------------------------------------------------------


class RealTimePolicyController:
    def __init__(
        self,
        xml_file,
        policy_path,
        device="cuda",
        record_video=False,
        record_proprio=False,
        headless=True,
    ):
        self.device = device
        self.headless = headless

        # Load policy
        self.session = ort.InferenceSession(policy_path)
        print(f"Policy loaded from {policy_path}")

        # Load the metadata
        self.model_metadata = self.session.get_modelmeta().custom_metadata_map
        print(f"ONNX metadata keys: {list(self.model_metadata.keys())}")
        self.model_metadata["joint_names"] = self.model_metadata["joint_names"].split(
            ","
        )
        self.model_metadata["joint_stiffness"] = [
            float(i) for i in self.model_metadata["joint_stiffness"].split(",")
        ]
        self.model_metadata["joint_damping"] = [
            float(i) for i in self.model_metadata["joint_damping"].split(",")
        ]
        self.model_metadata["default_joint_pos"] = [
            float(i) for i in self.model_metadata["default_joint_pos"].split(",")
        ]
        self.model_metadata["action_scale"] = [
            float(i) for i in self.model_metadata["action_scale"].split(",")
        ]

        # Get the reference motion length from ONNX constant node
        model = onnx.load(policy_path)
        for node in model.graph.node:
            if node.name == "/Constant":
                for attr in node.attribute:
                    if attr.name == "value":
                        self.length = onnx.numpy_helper.to_array(attr.t)
        print(f"Constant node found: value: {self.length}")

        # Create MuJoCo sim
        self.mj_model = mujoco.MjModel.from_xml_path(xml_file)
        self.mj_model.opt.timestep = 0.002  # the physics simulation runs at this time step
        self.data = mujoco.MjData(self.mj_model)

        # BM's 14 tracked bodies in ONNX output order (from BM config body_names)
        # These are the bodies the ONNX model outputs positions/quats for
        BM_BODY_NAMES = [
            'pelvis', 'left_hip_roll_link', 'left_knee_link', 'left_ankle_roll_link',
            'right_hip_roll_link', 'right_knee_link', 'right_ankle_roll_link',
            'torso_link', 'left_shoulder_roll_link', 'left_elbow_link',
            'left_wrist_yaw_link', 'right_shoulder_roll_link', 'right_elbow_link',
            'right_wrist_yaw_link',
        ]
        self.onnx_body_names = BM_BODY_NAMES
        self.num_onnx_bodies = len(BM_BODY_NAMES)
        print(f"ONNX outputs {self.num_onnx_bodies} bodies, MuJoCo has {self.mj_model.nbody - 1} bodies")
        print(f"BM body order: {self.onnx_body_names}")

        root_body = "pelvis"
        self.root_body_id = self.onnx_body_names.index(root_body)  # 0
        self.anchor_body_name = self.model_metadata.get("anchor_body_name", "torso_link")
        self.anchor_body_id = self.onnx_body_names.index("torso_link")  # 7

        self.robot_root_body_id = self.mj_model.body(root_body).id
        self.robot_anchor_body_id = self.mj_model.body(self.anchor_body_name).id

        # Build joint name -> param dicts from metadata
        joint_stiffness = dict(zip(self.model_metadata["joint_names"], self.model_metadata["joint_stiffness"]))
        joint_damping = dict(zip(self.model_metadata["joint_names"], self.model_metadata["joint_damping"]))

        # Set the parameters in the correct order
        model_joint_names = [
            self.mj_model.joint(i).name for i in range(1, self.mj_model.njnt)
        ]
        self.isaaclab2mujoco = [
            self.model_metadata["joint_names"].index(name) for name in model_joint_names
        ]
        self.mujoco2isaaclab = [
            model_joint_names.index(name) for name in self.model_metadata["joint_names"]
        ]
        stiffness = np.array([joint_stiffness[name] for name in model_joint_names])
        damping = np.array([joint_damping[name] for name in model_joint_names])
        default_dof_pos = np.array(self.model_metadata["default_joint_pos"])
        action_scale = np.array(self.model_metadata["action_scale"])

        if not self.headless:
            self.viewer = mjv.launch_passive(
                self.mj_model, self.data, show_left_ui=False, show_right_ui=False,
            )
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            # Consistent lighting across methods
            self.mj_model.vis.headlight.diffuse[:] = [0.6, 0.6, 0.6]
            self.mj_model.vis.headlight.ambient[:] = [0.15, 0.15, 0.15]
            self.mj_model.vis.headlight.specular[:] = [0.5, 0.5, 0.5]
            self.viewer.cam.distance = 3.0
            self.viewer.cam.elevation = -35

        # Example defaults & placeholders
        self.num_actions = 29
        self.sim_duration = 100000.0
        self.sim_dt = 0.002
        self.sim_decimation = 10  # the policy is called every 10 time steps

        # PD Gains, etc. (adapt as needed)
        self.default_dof_pos = default_dof_pos
        self.mujoco_default_dof_pos = np.concatenate(
            [np.array([0, 0, 0.793]), np.array([1, 0, 0, 0]), default_dof_pos]
        )
        self.stiffness = stiffness
        self.damping = damping
        self.torque_limits = np.array(
            [
                88,
                139,
                88,
                139,
                50,
                50,
                88,
                139,
                88,
                139,
                50,
                50,
                88,
                50,
                50,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
                25,
            ]
        )
        self.torque_limits[20] = 5
        self.torque_limits[27] = 5

        self.last_action = np.zeros(self.num_actions)

        self.action_scale = action_scale

        self.record_video = record_video
        self.record_proprio = record_proprio
        self.proprio_recordings = [] if record_proprio else None

    def extract_data(self):
        qpos = self.data.qpos.astype(np.float32)
        qvel = self.data.qvel.astype(np.float32)

        dof_pos = qpos[7:]
        dof_vel = qvel[6:]

        return dof_pos, dof_vel

    def reset_sim(self):
        mujoco.mj_resetData(self.mj_model, self.data)
        mujoco.mj_forward(self.mj_model, self.data)

    def reset(self, qpos, qvel):
        # body & hand
        self.data.qpos[:7] = qpos[:7]
        mujoco.mj_forward(self.mj_model, self.data)

    def get_state(self, time_step):
        """
        Returns the dof position and velocity at time step time_step.
        """
        # Offset by _start_step for combined motion models
        onnx_step = time_step + getattr(self, '_start_step', 0)
        # The ONNX policy has a time_step input and outputs joint_pos and joint_vel
        ort_inputs = {
            "obs": np.zeros(self.session.get_inputs()[0].shape, dtype=np.float32),
            "time_step": np.array([[onnx_step]], dtype=np.float32),
        }
        ort_outs = self.session.run(None, ort_inputs)

        output_names = [output.name for output in self.session.get_outputs()]
        body_pos_output_idx = output_names.index("body_pos_w")
        body_quat_output_idx = output_names.index("body_quat_w")
        joint_pos_output_idx = output_names.index("joint_pos")

        body_vel_output_idx = output_names.index("body_lin_vel_w")
        body_ang_vel_output_idx = output_names.index("body_ang_vel_w")
        joint_vel_output_idx = output_names.index("joint_vel")

        qpos = np.concatenate(
            [
                ort_outs[body_pos_output_idx].squeeze()[self.root_body_id],
                ort_outs[body_quat_output_idx].squeeze()[self.root_body_id],
                ort_outs[joint_pos_output_idx].squeeze()[self.isaaclab2mujoco],
            ]
        )

        qvel = np.concatenate(
            [
                ort_outs[body_vel_output_idx].squeeze()[self.root_body_id],
                ort_outs[body_ang_vel_output_idx].squeeze()[self.root_body_id],
                ort_outs[joint_vel_output_idx].squeeze()[self.isaaclab2mujoco],
            ]
        )
        return (qpos, qvel)

    def get_observation(self, time_step):
        # Offset by _start_step for combined motion models
        onnx_step = time_step + getattr(self, '_start_step', 0)
        # The ONNX policy has a time_step input and outputs joint_pos and joint_vel
        ort_inputs = {
            "obs": np.zeros(self.session.get_inputs()[0].shape, dtype=np.float32),
            "time_step": np.array([[onnx_step]], dtype=np.float32),
        }
        ort_outs = self.session.run(None, ort_inputs)
        output_names = [output.name for output in self.session.get_outputs()]
        joint_pos_output_idx = output_names.index("joint_pos")
        joint_vel_output_idx = output_names.index("joint_vel")

        body_pos_output_idx = output_names.index("body_pos_w")
        anchor_pos_w = torch.from_numpy(
            ort_outs[body_pos_output_idx].squeeze()[self.anchor_body_id],
        )
        body_quat_output_idx = output_names.index("body_quat_w")
        anchor_quat_w = torch.from_numpy(
            ort_outs[body_quat_output_idx].squeeze()[self.anchor_body_id],
        )
        anchor_pos_b = motion_anchor_pos_b(
            robot_anchor_pos_w=torch.from_numpy(
                self.data.xpos[self.robot_anchor_body_id]
            ),
            robot_anchor_quat_w=torch.from_numpy(
                self.data.xquat[self.robot_anchor_body_id]
            ),
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        )
        anchor_ori_b = motion_anchor_ori_b(
            robot_anchor_pos_w=torch.from_numpy(
                self.data.xpos[self.robot_anchor_body_id]
            ),
            robot_anchor_quat_w=torch.from_numpy(
                self.data.xquat[self.robot_anchor_body_id]
            ),
            anchor_pos_w=anchor_pos_w,
            anchor_quat_w=anchor_quat_w,
        )

        return torch.cat(
            [
                torch.from_numpy(
                    ort_outs[joint_pos_output_idx].squeeze()
                ),  # command joint position
                torch.from_numpy(
                    ort_outs[joint_vel_output_idx].squeeze()
                ),  # command joint velocity
                anchor_pos_b.squeeze(),  # position error in tracking the anchor
                anchor_ori_b.flatten(),  # orientation error in tracking the anchor
                base_lin_vel(
                    torch.from_numpy(self.data.xquat[1]),
                    torch.from_numpy(self.data.cvel[1, 3:]),  # self.data.qvel[:3]
                ),  # Root velocity in root frame
                base_ang_vel(
                    torch.from_numpy(self.data.xquat[1]),
                    torch.from_numpy(self.data.cvel[1, :3]),  # self.data.qvel[3:6]
                ),  # Root angular velocity in root frame
                joint_pos_rel(
                    torch.from_numpy(self.data.qpos[7:][self.mujoco2isaaclab]),
                    # this one is in the correct order for the policy. For MuJoCo
                    # we have the other one
                    np.array(self.model_metadata["default_joint_pos"]),
                ),
                torch.from_numpy(self.data.qvel[6:][self.mujoco2isaaclab]),
                torch.from_numpy(self.last_action),
            ],
            dim=-1,
        )

    def run(self, output_dir=None, motion_name="unknown", method_name="bm"):
        # Optionally record video
        mp4_writer = None
        renderer = None
        if self.record_video:
            import imageio
            video_name = os.path.join(output_dir or ".", f"bm_{motion_name}_sim2sim.mp4")
            print(f"Saving video to {video_name}")
            mp4_writer = imageio.get_writer(video_name, fps=50)

            if self.headless:
                # Use offscreen rendering
                renderer = mujoco.Renderer(self.mj_model, height=480, width=640)
                self.cam = mujoco.MjvCamera()
                self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self.cam.distance = 3.0
                self.cam.azimuth = 90
                self.cam.elevation = -20
            # else: use self.viewer.read_pixels()

        self.reset_sim()
        self.reset(*self.get_state(0))

        steps = int(self.sim_duration / self.sim_dt)
        pbar = tqdm(range(steps), desc="BM sim2sim eval")

        # ── Eval logging ──
        sim_body_pos_list = []
        ref_body_pos_list = []
        sim_joint_pos_list = []
        ref_joint_pos_list = []
        sim_qpos_list = []
        ref_qpos_list = []
        terminated_step = -1
        FALL_HEIGHT = 0.3
        all_body_ids = list(range(1, self.mj_model.nbody))
        all_body_names = [self.mj_model.body(i).name for i in all_body_ids]
        ref_data = mujoco.MjData(self.mj_model)  # separate data for FK on reference
        control_dt = self.sim_dt * self.sim_decimation

        try:
            for i in pbar:
                t_start = time.time()
                dof_pos, dof_vel = self.extract_data()

                if i // self.sim_decimation > self.length:
                    print("Reference motion finished.")
                    break

                if i % self.sim_decimation == 0:  # Query the policy
                    curr_step = i // self.sim_decimation

                    # Get reference state for this timestep
                    ref_qpos, ref_qvel = self.get_state(curr_step)

                    # Compute the final observation
                    obs_tensor = (
                        self.get_observation(curr_step)
                        .float()
                        .unsqueeze(0)
                    )

                    # Run inference
                    ort_inputs = {
                        "obs": obs_tensor.numpy(),
                        "time_step": np.array([[0]], dtype=np.float32),
                    }
                    raw_action = self.session.run(["actions"], ort_inputs)[0].squeeze()

                    # Store the raw action and process it
                    self.last_action = raw_action
                    scaled_actions = raw_action * self.action_scale
                    pd_target = scaled_actions + self.default_dof_pos

                    # Render / record video
                    pelvis_pos = self.data.xpos[self.mj_model.body("pelvis").id]
                    if not self.headless:
                        # Follow only XY, keep camera height fixed
                        self.viewer.cam.lookat[0] = pelvis_pos[0]
                        self.viewer.cam.lookat[1] = pelvis_pos[1]
                        self.viewer.cam.lookat[2] = 0.8  # fixed Z height
                        self.viewer.sync()
                        # Real-time pacing at control rate
                        elapsed = time.time() - t_start
                        target = control_dt
                        if elapsed < target:
                            time.sleep(target - elapsed)
                        if mp4_writer is not None:
                            img = self.viewer.read_pixels()
                            mp4_writer.append_data(img)
                    elif mp4_writer is not None and renderer is not None:
                        # Offscreen render: follow pelvis
                        self.cam.lookat[:] = pelvis_pos
                        renderer.update_scene(self.data, camera=self.cam)
                        img = renderer.render()
                        mp4_writer.append_data(img)

                    # ── Log body positions for comparison ──
                    # Reference body pos via FK
                    ref_data.qpos[:] = ref_qpos
                    mujoco.mj_forward(self.mj_model, ref_data)

                    sim_body_pos_list.append(self.data.xpos[all_body_ids].copy())
                    ref_body_pos_list.append(ref_data.xpos[all_body_ids].copy())
                    sim_joint_pos_list.append(dof_pos.copy())
                    ref_joint_pos_list.append(ref_qpos[7:].astype(np.float32).copy())
                    sim_qpos_list.append(self.data.qpos.copy())
                    ref_qpos_list.append(ref_qpos.copy())

                    # Fall detection
                    root_height = self.data.qpos[2]
                    if root_height < FALL_HEIGHT and terminated_step < 0:
                        terminated_step = curr_step
                        print(f"\n[WARN] Fall at step {curr_step}, height={root_height:.3f}")

                # PD control
                torque = (
                    pd_target[self.isaaclab2mujoco] - dof_pos
                ) * self.stiffness - dof_vel * self.damping
                torque = np.clip(torque, -self.torque_limits, self.torque_limits)

                # Apply the torques directly after fixing the order
                self.data.qfrc_applied[6:] = torque

                mujoco.mj_step(self.mj_model, self.data)

        except Exception as e:
            print(f"Error in run: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if mp4_writer is not None:
                mp4_writer.close()
                print("Video saved")

            if not self.headless:
                self.viewer.close()

        # ── Save eval log ──
        if output_dir and len(sim_body_pos_list) > 0:
            os.makedirs(output_dir, exist_ok=True)
            total_steps = len(sim_body_pos_list)
            out_path = os.path.join(output_dir, f"{method_name}_{motion_name}_log.npz")
            np.savez(
                out_path,
                ref_joint_pos=np.array(ref_joint_pos_list),
                sim_joint_pos=np.array(sim_joint_pos_list),
                ref_body_pos=np.array(ref_body_pos_list),
                sim_body_pos=np.array(sim_body_pos_list),
                sim_qpos=np.array(sim_qpos_list),
                ref_qpos=np.array(ref_qpos_list),
                dt=control_dt,
                terminated_step=terminated_step,
                total_steps=total_steps,
                body_names=np.array(all_body_names),
                method="bm_sim2sim",
            )
            print(f"\n[INFO] Saved eval log to {out_path}")
            print(f"  sim_body_pos: {np.array(sim_body_pos_list).shape}")
            print(f"  Success: {terminated_step < 0}")
            if terminated_step >= 0:
                print(f"  Survived: {terminated_step}/{total_steps} ({100*terminated_step/total_steps:.1f}%)")


def download_onnx_from_wandb(wandb_path, dest_dir):
    """Download ONNX from a wandb run. Returns path to downloaded file."""
    import wandb
    api = wandb.Api()

    run_path = wandb_path
    if "model" in wandb_path:
        run_path = "/".join(wandb_path.split("/")[:-1])

    run = api.run(run_path)
    onnx_files = [f.name for f in run.files() if f.name.endswith(".onnx")]
    if not onnx_files:
        raise FileNotFoundError(f"No .onnx files found in wandb run {run_path}")

    # Use specific file if specified, else latest
    if "model" in wandb_path:
        target = wandb_path.split("/")[-1]
    else:
        target = max(onnx_files, key=lambda x: x)

    print(f"Downloading {target} from {run_path}...")
    os.makedirs(dest_dir, exist_ok=True)
    run.file(target).download(dest_dir, replace=True)
    onnx_path = os.path.join(dest_dir, target)
    print(f"Saved to {onnx_path}")
    return onnx_path


def main_low_level_sim(args):
    # Resolve ONNX path
    onnx_path = args.onnx
    if onnx_path is None and args.wandb_path:
        dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "onnx")
        onnx_path = download_onnx_from_wandb(args.wandb_path, dest)
    if onnx_path is None:
        raise ValueError("Must provide --onnx or --wandb_path")

    # Resolve XML path
    xml_file = args.xml_file
    if xml_file is None:
        # Try to find a G1 MuJoCo XML
        HERE = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            # BM's own G1 XML (best match for BM policy)
            os.path.join(HERE, "..", "source", "whole_body_tracking", "whole_body_tracking", "assets", "unitree_description", "mjcf", "g1.xml"),
            os.path.join(HERE, "..", "..", "..", "TWIST2", "assets", "g1", "g1_sim2sim_29dof.xml"),
            os.path.join(HERE, "..", "..", "..", "humanoid-general-motion-tracking", "assets", "robots", "g1", "g1.xml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                xml_file = os.path.abspath(c)
                print(f"Auto-detected XML: {xml_file}")
                break
        if xml_file is None:
            raise ValueError("Must provide --xml_file or have a G1 XML in the repo")

    # Handle start/end step for combined motion ONNX models
    start_step = args.start_step
    end_step = args.end_step

    controller = RealTimePolicyController(
        xml_file=xml_file,
        policy_path=onnx_path,
        device="cuda",
        record_video=args.record_video,
        headless=args.headless,
    )

    # Override length if end_step is set
    if end_step is not None and start_step is not None:
        controller.length = float(end_step - start_step)
        controller._start_step = start_step
    else:
        controller._start_step = 0

    controller.run(output_dir=args.output_dir, motion_name=args.motion_name, method_name=args.method_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BM sim2sim eval in MuJoCo")
    HERE = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--onnx", default=None, help="Path to the ONNX model (if already downloaded)")
    parser.add_argument("--wandb_path", default=None, help="Wandb run path to download ONNX from (e.g. berkeley-humanoid/lafan_full/34id9czx)")
    parser.add_argument("--xml_file", default=None, help="MuJoCo XML file (auto-detected if not provided)")
    parser.add_argument("--output_dir", default="/move/u/takaraet/rebuttal/eval_logs", help="Directory to save eval logs")
    parser.add_argument("--motion_name", default="dance1_subject1", help="Motion name for output file")
    parser.add_argument("--start_step", type=int, default=None, help="Start ONNX timestep (for combined motion models)")
    parser.add_argument("--end_step", type=int, default=None, help="End ONNX timestep (for combined motion models)")
    parser.add_argument("--list_motions", action="store_true", help="List walk_combined motion boundaries and exit")
    parser.add_argument("--record_video", action="store_true", help="Record a video")
    parser.add_argument("--headless", action="store_true", default=False, help="Run in headless mode")
    parser.add_argument("--method_name", default="bm", help="Method name prefix for output files (default: bm)")
    args = parser.parse_args()

    # Handle --list_motions
    if args.list_motions:
        print("Walk-combined ONNX motion boundaries:")
        for name, (start, end) in WALK_COMBINED_MOTIONS.items():
            dur = (end - start) * 0.02
            print(f"  {name}: steps {start} - {end} ({dur:.1f}s)")
        print(f"\nUsage: --start_step <start> --end_step <end> --motion_name <name>")
        exit(0)

    # Auto-resolve motion name to start/end for walk_combined
    if args.start_step is None and args.motion_name in WALK_COMBINED_MOTIONS:
        args.start_step, args.end_step = WALK_COMBINED_MOTIONS[args.motion_name]
        print(f"Auto-resolved {args.motion_name}: steps {args.start_step} - {args.end_step}")

    main_low_level_sim(args)
