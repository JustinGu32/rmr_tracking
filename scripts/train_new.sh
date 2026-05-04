#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --gres=gpu:titanrtx:1 
#SBATCH --job-name=crane_new
#SBATCH --output=logs/slurm/crane_new_%j.out
#SBATCH --error=logs/slurm/crane_new_%j.err

set -euo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/rsl_rl/train_bones.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane_new:v0 \
  --headless \
  --logger wandb \
  --log_project_name multiclip_bones_popart \
  --run_name crane_new_tracking_baseline \
  --video \
  --video_length 500 \
  --video_interval 10000

# --task=Bones-Flat-chip-G1-v0 \
#   --popart_multihead \
#   --popart_head_mode grouped \
#   --popart_group_preset actual_individual