#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:a5000:1 
#SBATCH --job-name=takara_walk_isaac

set -euo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# export WANDB_ENTITY=kkarenvoo

python scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name justingu-stanford-university-org/wandb-registry-motions/takara_walk_isaac:v0 \
  --headless \
  --logger wandb \
  --log_project_name takara_walk_isaac \
  --run_name takara_walk_isaac_npz \
  --video \
  --video_length 500 \
  --video_interval 10000 \
  --max_iterations 30000

