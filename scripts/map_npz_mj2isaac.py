import numpy as np
import sys
from pathlib import Path

mujoco_npz_path = Path("/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase.npz")
out_path = Path("/move/u/karenvo/Projects/rmr_tracking/artifacts/staircase_2.npz")


isaac_joint_names = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint']

# Source joint ordering from MuJoCo
mujoco_joint_names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"]
mujoco_npz = np.load(mujoco_npz_path)

# Map MuJoCo indices to IsaacLab indices
joint_mapping = [mujoco_joint_names.index(name) for name in isaac_joint_names]


isaac_body_names = ['pelvis', 'left_hip_pitch_link', 'right_hip_pitch_link', 'waist_yaw_link', 'left_hip_roll_link', 'right_hip_roll_link', 'waist_roll_link', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_knee_link', 'right_knee_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_ankle_pitch_link', 'right_ankle_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_elbow_link', 'right_elbow_link', 'left_wrist_roll_link', 'right_wrist_roll_link', 'left_wrist_pitch_link', 'right_wrist_pitch_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link']
mujoco_body_names = ['pelvis', 'left_hip_pitch_link', 'left_hip_roll_link', 'left_hip_yaw_link', 'left_knee_link', 'left_ankle_pitch_link', 'left_ankle_roll_link', 'right_hip_pitch_link', 'right_hip_roll_link', 'right_hip_yaw_link', 'right_knee_link', 'right_ankle_pitch_link', 'right_ankle_roll_link', 'waist_yaw_link', 'waist_roll_link', 'torso_link', 'left_shoulder_pitch_link', 'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'left_wrist_roll_link', 'left_wrist_pitch_link', 'left_wrist_yaw_link', 'right_shoulder_pitch_link', 'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link', 'right_wrist_roll_link', 'right_wrist_pitch_link', 'right_wrist_yaw_link']

body_mapping = [mujoco_body_names.index(name) for name in isaac_body_names]
fps = mujoco_npz["fps"]
# If this file’s "joint_pos" is actually qpos (has root free joint), strip it.
jp = mujoco_npz["joint_pos"]
jv = mujoco_npz["joint_vel"]

# 29 joints. qpos has 7 root (3 pos + 4 rot), qvel has 6 root (3 lin + 3 ang)
if jp.shape[1] >= 7 + 29:
    jp = jp[:, 7:7+29]
if jv.shape[1] >= 6 + 29:
    jv = jv[:, 6:6+29]

joint_pos = jp[:, joint_mapping]
joint_vel = jv[:, joint_mapping]
body_pos_w = mujoco_npz["body_pos_w"][:, body_mapping]
body_quat_w = mujoco_npz["body_quat_w"][:, body_mapping]
body_lin_vel_w = mujoco_npz["body_lin_vel_w"][:, body_mapping]
body_ang_vel_w = mujoco_npz["body_ang_vel_w"][:, body_mapping]

arr = {
    "fps": fps,
    "joint_pos": joint_pos,
    "joint_vel": joint_vel,
    "body_pos_w": body_pos_w,
    "body_quat_w": body_quat_w,
    "body_lin_vel_w": body_lin_vel_w,
    "body_ang_vel_w": body_ang_vel_w,
    "joint_names": isaac_joint_names,
}

if "object_pos_w" in mujoco_npz:
    arr["object_pos_w"] = mujoco_npz["object_pos_w"]
if "object_quat_w" in mujoco_npz:
    arr["object_quat_w"] = mujoco_npz["object_quat_w"]
if "object_lin_vel_w" in mujoco_npz:
    arr["object_lin_vel_w"] = mujoco_npz["object_lin_vel_w"]
if "object_ang_vel_w" in mujoco_npz:
    arr["object_ang_vel_w"] = mujoco_npz["object_ang_vel_w"]

np.savez(out_path, **arr)

print(f"Saved {out_path}")
