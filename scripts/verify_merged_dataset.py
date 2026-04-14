import zarr
import numpy as np
import argparse

def verify_merged_dataset(dataset_path):
    print(f"Verifying {dataset_path}...")
    
    try:
        root = zarr.open(dataset_path, mode='r')
    except Exception as e:
        print(f"[ERROR] Could not open dataset: {e}")
        return

    # 1. Check groups existence
    if 'data' not in root or 'meta' not in root:
        print("[ERROR] Missing 'data' or 'meta' group.")
        return
    
    data = root['data']
    meta = root['meta']

    # 2. Check Episode Ends
    if 'episode_ends' not in meta:
        print("[ERROR] 'episode_ends' not found in meta.")
        return
    
    episode_ends = meta['episode_ends'][:]
    print(f"Episode Ends Length: {len(episode_ends)}")
    print(f"Episode Ends (first 5): {episode_ends[:5]}")
    print(f"Episode Ends (last 5): {episode_ends[-5:]}")

    # Check monotonicity
    if not np.all(np.diff(episode_ends) > 0):
         print("[ERROR] 'episode_ends' is NOT strictly increasing!")
    else:
         print("[PASS] 'episode_ends' is strictly increasing.")

    # 3. Check Data Consistency
    if 'act' not in data:
         print("[WARNING] 'act' not found in data. Cannot check length against episode_ends.")
    else:
        act_len = data['act'].shape[0]
        last_end = episode_ends[-1]
        
        print(f"Data ('act') Length: {act_len}")
        print(f"Last Episode End: {last_end}")
        
        if act_len == last_end:
            print("[PASS] Data length matches last episode end.")
        else:
            print(f"[ERROR] Mismatch! Data length {act_len} != Last episode end {last_end}")

    # 4. Check num_episodes
    if 'num_episodes' in meta:
        num_eps = meta['num_episodes'][0] if isinstance(meta['num_episodes'], zarr.core.Array) else meta['num_episodes']
        if num_eps == len(episode_ends):
             print(f"[PASS] num_episodes ({num_eps}) matches len(episode_ends).")
        else:
             print(f"[ERROR] num_episodes ({num_eps}) does NOT match len(episode_ends) ({len(episode_ends)}).")

    print("\nVerification Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    args = parser.parse_args()
    
    verify_merged_dataset(args.dataset_path)
