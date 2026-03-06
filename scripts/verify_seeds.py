import zarr
import numpy as np
import glob
import os

def check_seeds():
    # Find zarr files
    files = sorted(glob.glob("VISION_OBSTACLE_DATA_COLLECTED/*.zarr"))
    # Filter out merged_dataset.zarr
    files = [f for f in files if "merged_dataset" not in f]
    
    if len(files) < 2:
        print("Need at least 2 datasets to compare.")
        return

    print(f"Comparing {files[0]} \nand {files[1]}")

    z1 = zarr.open(files[0], mode='r')
    z2 = zarr.open(files[1], mode='r')

    # Compare root position of first few steps
    # data/root_pos
    
    d1 = z1['data/root_pos'][:]
    d2 = z2['data/root_pos'][:]
    
    # Check for exact equality
    if np.array_equal(d1, d2):
        print("\n[WARNING] Datasets are IDENTICAL! Seeds might not be working.")
    else:
        # Check how different they are
        diff = np.abs(d1 - d2).mean()
        print(f"\n[SUCCESS] Datasets are DIFFERENT.")
        print(f"Average element-wise difference in root_pos: {diff}")
        
        # Check first step specifically (might be same if restart is deterministic before physics)
        # usually physics init takes one step to diverge if seeded differently?
        # or reset() places them differently if random initialization is on.
        
        print("First step d1:", d1[0])
        print("First step d2:", d2[0])
        print("First step equal?", np.array_equal(d1[0], d2[0]))

if __name__ == "__main__":
    check_seeds()
