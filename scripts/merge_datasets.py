"""
python scripts/merge_datasets.py \
    --input_dir VISION_OBSTACLE_DATA_COLLECTED \
    --output_path VISION_OBSTACLE_DATA_COLLECTED/merged_dataset.zarr \
    --pattern "*.zarr"
"""

import zarr
import argparse
from pathlib import Path
import os
import shutil
from tqdm import tqdm
import numpy as np

def merge_zarr_datasets(input_dir, output_path, pattern="*.zarr"):
    """
    Merges multiple Zarr datasets (ReplayBuffer style) into a single Zarr archive.
    Structure assumed:
      - data/ (contains concatenated arrays like act, obs, etc.)
      - meta/ (contains episode_ends, etc.)
    """
    input_path = Path(input_dir)
    output_path = Path(output_path)
    
    # Find all zarr files
    zarr_files = sorted(list(input_path.glob(pattern)))
    if not zarr_files:
        print(f"No Zarr files found in {input_dir} matching {pattern}")
        return

    print(f"Found {len(zarr_files)} datasets to merge.")
    
    # Create the output directory/group
    if output_path.exists():
        print(f"Output path {output_path} already exists. Overwriting/Appending.")
        # shutil.rmtree(output_path) # Optional: clear if you want fresh start
    
    store = zarr.DirectoryStore(str(output_path))
    root = zarr.group(store=store, overwrite=True)
    
    # Initialize groups
    data_group = root.create_group('data', overwrite=True)
    meta_group = root.create_group('meta', overwrite=True)
    
    total_steps = 0
    total_episodes = 0
    all_episode_ends = []
    
    # We need to accumulate data arrays. 
    # Since Zarr arrays are on disk, we can append to them efficiently.
    # We'll initialize them with the first file.
    
    first_file = True
    
    for zf_path in tqdm(zarr_files, desc="Merging datasets"):
        try:
            # Open source dataset
            with zarr.open_group(str(zf_path), mode='r') as src:
                
                # Check for empty dataset or missing groups
                if 'data' not in src or 'meta' not in src:
                    print(f"Skipping {zf_path}: Missing data/ or meta/ group")
                    continue
                
                src_data = src['data']
                src_meta = src['meta']
                
                # Get current file's length (timesteps)
                # We can check any array in data, e.g., 'act'
                if 'act' not in src_data:
                     print(f"Skipping {zf_path}: Missing 'act' array")
                     continue
                
                current_file_steps = src_data['act'].shape[0]
                if current_file_steps == 0:
                    continue

                # --- Handle Data Arrays ---
                for key in src_data.keys():
                    arr = src_data[key]
                    if first_file:
                        # Create array in output with same shape/dtype, but extensible 0-th dim
                        shape = list(arr.shape)
                        chunks = list(arr.chunks)
                        # Ensure 0-th dim is chunked reasonable? auto is usually fine.
                        
                        # creating extensible array
                        # We initialize with data from first file
                        init_data = arr[:] # Load to memory (assuming fits in memory)
                        
                        # Create array
                        ds = data_group.create_dataset(
                            key, 
                            data=init_data, 
                            shape=shape, 
                            chunks=chunks, 
                            dtype=arr.dtype,
                            compressor=arr.compressor,
                            fill_value=arr.fill_value
                        )
                    else:
                        # Append to existing array
                        if key in data_group:
                            ds = data_group[key]
                            # verify shapes match (except dim 0)
                            if ds.shape[1:] != arr.shape[1:]:
                                print(f"Warning: Shape mismatch for {key} in {zf_path}. Skipping array.")
                                continue
                            
                            ds.append(arr[:])
                        else:
                            print(f"Warning: New key {key} found in {zf_path} but not in first file. Skipping.")

                # --- Handle Metadata ---
                # Specifically 'episode_ends' needs offset adjustment
                # We need to extract the raw values, add current total_steps, then append.
                if 'episode_ends' in src_meta:
                    ends = src_meta['episode_ends'][:] 
                    
                    # Add offset to these ends so they point to the correct index in the concatenated array
                    ends_shifted = ends + total_steps
                    all_episode_ends.append(ends_shifted)
                
                # Other metadata (scalars or small arrays)
                # We mainly care about accumulating counts or keeping first file's config
                
                # Update totals
                total_steps += current_file_steps
                
                # Track episode count from metadata if available, or infer from len(episode_ends)
                # Usually num_episodes is a scalar array in meta
                # We can verify at the end.
                
                first_file = False
                
        except Exception as e:
            print(f"Error processing {zf_path}: {e}")
            import traceback
            traceback.print_exc()

    # Finalize Metadata
    # Write aggregated episode_ends as a single non-chunked array (or single chunk)
    if all_episode_ends:
        final_episode_ends = np.concatenate(all_episode_ends)
        # chunks=False ensures it's written as a single block
        meta_group.create_dataset('episode_ends', data=final_episode_ends, chunks=False, overwrite=True)
        
        final_eps_count = len(final_episode_ends)
        meta_group.create_dataset('num_episodes', data=np.array([final_eps_count], dtype=np.int64), overwrite=True)
        print(f"Total episodes merged: {final_eps_count}")
    
    # Save total steps?
    # Usually strictly not needed if we have array lengths, but good for consistency
    # meta/collect_steps might be per-episode or total? In rsl_rl it usually means *requested* steps.
    
    # Copy other static metadata from first file (args, config)
    # We copy everything from the first file's meta group that isn't 'episode_ends' or 'num_episodes'
    # Use the LAST file processed (src is closed, but we can re-open first or just use aggregated dict if we saved it)
    
    # Let's re-open the first file to copy static metadata
    if len(zarr_files) > 0:
        with zarr.open_group(str(zarr_files[0]), mode='r') as src:
            if 'meta' in src:
                for k, v in src['meta'].items():
                    if k not in ['episode_ends', 'num_episodes']:
                        # Copy parameter arrays/scalars
                        # If it's a group, copy group? RSL-RL usually has flat params in meta
                        # v is a dataset or group
                        if isinstance(v, zarr.core.Array):
                             # Check if we already created it (unlikely for static params unless we looped)
                             if k not in meta_group:
                                 # Use [...] to read content safely for both scalar and n-dim arrays
                                 meta_group.create_dataset(k, data=v[...], overwrite=True)
    
    print(f"Successfully merged datasets into {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple Zarr datasets.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing Zarr files.")
    parser.add_argument("--output_path", type=str, required=True, help="Path for the merged Zarr file.")
    parser.add_argument("--pattern", type=str, default="*.zarr", help="Glob pattern for Zarr files.")
    
    args = parser.parse_args()
    
    merge_zarr_datasets(args.input_dir, args.output_path, args.pattern)
