#!/bin/bash
#SBATCH --partition=move --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=collect_multiclip
#SBATCH --output=logs/slurm/collect_multiclip_%j.out
#SBATCH --error=logs/slurm/collect_multiclip_%j.err

set -euo pipefail

mkdir -p logs/slurm
cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
# export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
# export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run: c15qko8c = bones_target_33hz_swish_uniform_timetolive_walk-jog
# python scripts/rsl_rl/collect_dataset_multiclip.py \
#     --task=Bones-MultiClip-Compliance-G1-Collect-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --include_motion_types walk,jog \
#     --wandb_path=robot-mcrobotface/multiclip_bones/c15qko8c \
#     --activation swish \
#     --decimation 6 \
#     --num_envs 1 \
#     --num_steps_collect 60 \
#     --num_eps_collect 500 \
#     --episode_collect_length 4 \
#     --min_delay 0 \
#     --max_delay 0 \
#     --save_folder=./datasets \
#     --enable_cameras \
#     --headless

# ENABLE_CAMERAS=1 python scripts/rsl_rl/collect_dataset_multiclip.py \
python scripts/rsl_rl/collect_dataset_multiclip.py \
    --task=Bones-MultiClip-Compliance-G1-Collect-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
    --include_motion_types walk,jog \
    --wandb_path=robot-mcrobotface/multiclip_bones/dpm67awe \
    --activation swish \
    --decimation 6 \
    --num_envs 100 \
    --num_steps_collect 18 \
    --num_eps_collect 10000 \
    --episode_collect_length 8 \
    --save_folder=./datasets/walk_jog_33hz_dataset \
    --headless \
    --enable_cameras
