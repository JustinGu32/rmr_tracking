#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --job-name=chair_step

set -euo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

export WANDB_ENTITY=robot-mcrobotface

python scripts/rsl_rl/train.py \
   --task=Chair-Step-G1-v0 \
   --registry_name robot-mcrobotface/csv_to_npz/chair_step_truncated_converted.npz:latest \
   --headless \
   --logger wandb \
   --log_project_name chair_step \
   --run_name chair_step_v0 \
   --video \
   --video_length 500 \
   --video_interval 10000
