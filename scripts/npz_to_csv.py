
import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="Convert motion NPZ file to CSV.")
    parser.add_argument("--input_npz", type=str, required=True, help="Input NPZ file path.")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV file path.")
    args = parser.parse_args()

    if not os.path.exists(args.input_npz):
        print(f"Error: Input file {args.input_npz} does not exist.")
        return

    data = np.load(args.input_npz)

    # import pdb; pdb.set_trace()
    
    # Check for required keys
    # Isaac Sim / RSL RL motion format usually has:
    # 'root_pos' or 'body_pos_w' (take index 0 for root)
    # 'root_rot' or 'body_quat_w' (take index 0 for root)
    # 'joint_pos'
    
    keys = list(data.keys())
    print(f"Loading data from {args.input_npz}. Keys: {keys}")
    
    # Extract Root Position
    if 'root_pos' in data:
         root_pos = data['root_pos']
    elif 'body_pos_w' in data:
         # Root is at index 1
         root_pos = data['body_pos_w'][:, 1, :].copy()
    else:
         print("Error: Could not find root position data.")
         return

    # Extract Root Rotation
    if 'root_rot' in data:
         root_rot = data['root_rot']


    elif 'body_quat_w' in data:
         # Root is at index 1
         root_rot = data['body_quat_w'][:, 1, :]
    else:
         print("Error: Could not find root rotation data.")
         return
         
    # Ensure rotation is w, x, y, z (Isaac convention)
    # csv_to_npz expects [x, y, z, w] in CSV and converts to [w, x, y, z]
    # So we should save as [x, y, z, w]
    
    # Assuming standard Isaac [w, x, y, z]
    root_rot_xyzw = root_rot[:, [1, 2, 3, 0]]
    
    # Extract Joint Positions
    if 'joint_pos' in data:
        joint_pos = data['joint_pos']
    elif 'qpos' in data:
         print("Warning: Using 'qpos', verify if this includes root.")
         joint_pos = data['qpos']
    else:
         print("Error: Could not find joint position data.")
         return

    # Target Joint Names (G1)
    target_joints = [
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
            "right_wrist_yaw_joint"
    ]
    
    final_joint_pos = joint_pos
    
    # If joint_names available, reorder/filter
    if 'joint_names' in data:
        src_names = data['joint_names']
        if isinstance(src_names, np.ndarray):
             src_names = src_names.tolist()
        
        # Clean names
        src_names = [n.decode('utf-8') if isinstance(n, bytes) else str(n) for n in src_names]
        
        print(f"Found {len(src_names)} joints in NPZ.")
        
        indices = []
        missing = []
        for name in target_joints:
             if name in src_names:
                  indices.append(src_names.index(name))
             else:
                  # Try relaxed matching?
                  missing.append(name)
        
        if missing:
             print(f"Warning: {len(missing)} target joints not found in NPZ: {missing}")
             if len(indices) == 0:
                  print("No matching joints found. Using raw joint_pos (truncated if needed).")
             else:
                  final_joint_pos = joint_pos[:, indices]
        else:
             print("All target joints matched successfully.")
             final_joint_pos = joint_pos[:, indices]
    else:
         print("Warning: 'joint_names' not found in NPZ. Utilizing raw joint_pos assumes correct order.")
         # Check size
         if joint_pos.shape[1] != len(target_joints):
              print(f"Warning: Size mismatch! NPZ has {joint_pos.shape[1]}, target has {len(target_joints)}.")

    # Combine: [pos(3), rot(4), joints(N)]
    # Shape: (Frames, 7 + N)
    
    print(f"Shapes -- Pos: {root_pos.shape}, Rot: {root_rot_xyzw.shape}, Joints: {final_joint_pos.shape}")
    
    # Extract Object Data
    object_data = []
    has_object_data = False
    
    if 'object_pos_w' in data and 'object_quat_w' in data:
         object_pos = data['object_pos_w']
         object_rot = data['object_quat_w']
         
         # Convert rot to x, y, z, w if needed?
         # Isaac usually expects [w, x, y, z] for quat in code, but CSV convention might be different
         # Based on root_rot above: "csv_to_npz expects [x, y, z, w] in CSV and converts to [w, x, y, z]"
         # So we save as [x, y, z, w]
         object_rot_xyzw = object_rot[:, [1, 2, 3, 0]]
         
         print(f"Found object data. Pos: {object_pos.shape}, Rot: {object_rot_xyzw.shape}")
         object_data = [object_pos, object_rot_xyzw]
         has_object_data = True
    else:
         print("No object data found.")

    # Concatenate
    if has_object_data:
        motion_data = np.concatenate([root_pos, root_rot_xyzw, final_joint_pos] + object_data, axis=1)
    else:
        motion_data = np.concatenate([root_pos, root_rot_xyzw, final_joint_pos], axis=1)
    
    # Save CSV
    print(f"Saving to {args.output_csv}...")
    np.savetxt(args.output_csv, motion_data, delimiter=",")
    print("Done.")

if __name__ == "__main__":
    main()
