#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --job-name=relabel_vision
#SBATCH --output=logs/slurm/relabel_vision_stairs_%j.out

set -uo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/rsl_rl/relabel_vision_stairs.py \
    --task=Tracking-Flat-G1-Collect-v0 \
    --new_zarr_path=/move/u/karenvo/Projects/rmr_tracking/data/relabel_with_stairs_vision_flipped.zarr \
    --with_stairs \
    --staircase_ahead_m=0.15 \
    --staircase_yaw_bias_deg=0 \
    --cutoff_margin=0.35 \
    --headless

# python scripts/rsl_rl/relabel_vision_stairs.py \
#     --task=Tracking-Flat-G1-Collect-v0 \
#     --new_zarr_path=/move/u/karenvo/Projects/rmr_tracking/data/relabel_no_stairs_vision_flipped.zarr \
#     --without_stairs \
#     --staircase_ahead_m=0.15 \
#     --staircase_yaw_bias_deg=0 \
#     --cutoff_margin=0.35 \
#     --headless
