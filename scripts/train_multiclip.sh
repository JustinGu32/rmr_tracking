#!/bin/bash
#SBATCH --job-name=multiclip_train
#SBATCH --partition=humanoid  --account=move
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
# conda run -n rmr --live-stream python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --decimation=6 \
#     --future_steps=2,4,6,8 \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_tracking \
#     --run_name locomotion_33hz_future_2_4_6_8


# conda run -n rmr --live-stream python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --decimation=6 \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_tracking \
#     --run_name locomotion_33hz_resumed \
#     --wandb_resume=takaraet/multiclip_tracking/gqw88r28


# conda run -n rmr --live-stream python scripts/rsl_rl/train.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --decimation=4 \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_tracking \
#     --run_name locomotion_50hz_resumed \
#     --wandb_resume=takaraet/multiclip_tracking/ilu3k71i



conda run -n rmr --live-stream python scripts/rsl_rl/train.py \
    --task=Tracking-MultiClip-Flat-G1-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
    --decimation=4 \
    --num_envs=8192 \
    --num_steps_per_env=12 \
    --headless \
    --logger wandb \
    --log_project_name multiclip_tracking \
    --run_name locomotion_50hz_8k_12steps_swish_clipPhaseCriticOnly_adaptive \

