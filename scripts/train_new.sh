#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --gres=gpu:a5000:1 
#SBATCH --job-name=crane_new
#SBATCH --output=logs/slurm/crane_new_%j.out
#SBATCH --error=logs/slurm/crane_new_%j.err

set -euo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/rsl_rl/train_bones.py \
  --task=Bones-Flat-G1-v0 \
  --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane_new:v0 \
  --headless \
  --logger wandb \
  --log_project_name multiclip_bones_popart \
  --run_name crane_upperlower_raw_uniform_mom0.0005_v2 \
  --video \
  --video_length 500 \
  --video_interval 10000 \
  --popart_multihead \
  --popart_head_mode grouped \
  --popart_group_preset upper_lower \
  --popart_actor_advantage_scaling raw \
  --popart_grouped_actor_weight_mode uniform \
  --popart_momentum 0.0005 \
  --seed 1

# python scripts/rsl_rl/train_bones.py \
#   --task=Tracking-Flat-G1-v0 \
#   --registry_name justingu-stanford-university-org/wandb-registry-Motions/crane_new:v0 \
#   --headless \
#   --logger wandb \
#   --log_project_name multiclip_bones_popart \
#   --run_name crane_new_tracking_baseline \
#   --video \
#   --video_length 500 \
#   --video_interval 100