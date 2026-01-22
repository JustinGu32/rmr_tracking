import numpy as np
import sys
from pathlib import Path

npy_path = Path("/move/u/justingu/whole_body_tracking/artifacts/chair_step_Skeleton_004:v0/chair_step_Skeleton_004.npy")
out_path = Path("/move/u/justingu/whole_body_tracking/artifacts/chair_step_Skeleton_004:v0/motion.npz")

arr = np.load(npy_path)
np.savez(out_path, data=arr)

print(f"Saved {out_path}")
