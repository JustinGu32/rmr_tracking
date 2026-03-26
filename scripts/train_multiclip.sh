#!/bin/bash
#SBATCH --job-name=multiclip_train
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/multiclip_%j.out
#SBATCH --error=logs/slurm/multiclip_%j.err

# Create log directory
mkdir -p logs/slurm

# Activate conda
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate rmr

# # Run training
# cd /move/u/takaraet/rmr_tracking
conda run -n rmr --live-stream python scripts/rsl_rl/train.py \
    --task=Tracking-MultiClip-Flat-G1-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
    --num_envs=4096 \
    --headless \
    --logger wandb \
    --log_project_name multiclip_tracking \
    --run_name locomotion_33hz

