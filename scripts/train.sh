#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gres=gpu:rtxpro6000:1 
#SBATCH --job-name=staircase_compliance
#SBATCH --output=slurm_outputs/slurm-%A_%a.out


set -euo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# python scripts/rsl_rl/train.py \
#   --task=Tracking-Flat-G1-v0 \
#   --registry_name justingu-stanford-university-org/wandb-registry-motions/takara_walk_isaac:v0 \
#   --headless \
#   --logger wandb \
#   --log_project_name takara_walk_isaac \
#   --run_name takara_walk_isaac_npz \
#   --video \
#   --video_length 500 \
#   --video_interval 10000


python scripts/rsl_rl/train.py \
   --task=Staircase-G1-Compliance-v0 \
   --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
   --headless \
   --logger wandb \
   --log_project_name staircase \
   --run_name staircase_compliance_v0 \
   --video \
   --video_length 500 \
   --video_interval 10000
