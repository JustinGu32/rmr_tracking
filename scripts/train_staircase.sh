#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:rtxpro6000:1
#SBATCH --job-name=staircase

set -euo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Prevent NCCL conflicts with PhysX CUDA memory
# P2P disable: no direct GPU-to-GPU memory access
# SHM disable: no shared memory (still uses cudaMemcpy) — forces socket transport
# IB disable: no InfiniBand (not needed on single node)
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

export WANDB_ENTITY=robot-mcrobotface

python -m torch.distributed.run --nproc_per_node=2 scripts/rsl_rl/train.py \
   --task=Staircase-G1-Compliance-v0 \
   --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
   --headless \
   --logger wandb \
   --log_project_name staircase \
   --run_name staircase_lesserxy_pelvis_maxz_torque \
   --video \
   --video_length 500 \
   --video_interval 10000
