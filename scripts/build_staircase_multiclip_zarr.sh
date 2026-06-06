#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=build_staircase_multiclip_zarr
#SBATCH --output=logs/slurm/build_staircase_multiclip_zarr_%j.out

set -uo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/build_staircase_multiclip_zarr.py \
  --npz artifacts/walk_up_33.npz:v0/motion.npz \
  --npz artifacts/walk_down_33.npz:v0/motion.npz \
  --npz artifacts/up_continuous_33.npz:v0/motion.npz \
  --npz artifacts/down_continuous_33.npz:v0/motion.npz \
  --metadata artifacts/walk_up_karen_stairs/staircase_metadata.json \
  --metadata artifacts/walk_down_karen_stairs/staircase_metadata.json \
  --metadata artifacts/up_continuous_v2_karen_stairs/staircase_metadata.json \
  --metadata artifacts/down_continuous_v2_karen_stairs/staircase_metadata.json \
  --staircase-pos 3.2 -0.2 0.0 \
  --staircase-pos 3.2 1.95 0.0 \
  --staircase-pos 3.7 0.55 0.0 \
  --staircase-pos 3.65 0.47 0.0 \
  --staircase-quat 0.6819983600624985 0.0 0.0 0.7313537016191705 \
  --staircase-quat 0.6819983600624985 0.0 0.0 0.7313537016191705 \
  --staircase-quat 0.6819983600624985 0.0 0.0 0.7313537016191705 \
  --staircase-quat 0.6819983600624985 0.0 0.0 0.7313537016191705 \
  --output-zarr data/staircase_multiclip_33hz.zarr
