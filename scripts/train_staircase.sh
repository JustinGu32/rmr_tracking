#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=staircase_multiclip
#SBATCH --output=logs/slurm/staircase_multiclip_%j.out

set -euo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Prevent NCCL conflicts with PhysX CUDA memory
# P2P disable: no direct GPU-to-GPU memory access
# SHM disable: no shared memory (still uses cudaMemcpy) — forces socket transport
# IB disable: no InfiniBand (not needed on single node)
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_DEBUG=INFO

export WANDB_ENTITY=robot-mcrobotface

# python -m torch.distributed.run --nproc_per_node=2 scripts/rsl_rl/train.py \
# python -m torch.distributed.run --nproc_per_node=2 scripts/rsl_rl/train.py \
#    --task=Staircase-G1-v0 \
#    --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase \
#    --run_name staircase_high_stiff_2gpu \
#    --video \
#    --video_length 500 \
#    --video_interval 10000

# walk up: 3.2, -0.2
# walk down: 3.2, 1.95
# up: 3.7, 0.55
# down: 3.65, 0.47

python scripts/rsl_rl/train_stairs.py \
   --task=Staircase-G1-v0 \
   --registry_name robot-mcrobotface/csv_to_npz/walk_up_33.npz:latest \
   --depth_obs \
   --depth_cnn \
   --headless \
   --logger wandb \
   --log_project_name new_staircase \
   --run_name walk_up_karen_staircase_33hz_depth_headcam_delay3_depcnn \
   --video \
   --video_length 500 \
   --video_interval 10000

# python scripts/rsl_rl/train_stairs.py \
#    --task=Staircase-MultiClip-G1-v0 \
#    --zarr_path=data/staircase_multiclip_33hz.zarr \
#    --sampling uniform \
#    --use_depth \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase_multiclip \
#    --run_name staircase_multiclip_depth_uniform_decimation6_headcam_nodelay \
#    --video \
#    --video_length 500 \
#    --video_interval 10000
