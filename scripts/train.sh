#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:a5000:1 
#SBATCH --job-name=chip_tango_0.7m_dec4_2step
#SBATCH --output=slurm_outputs/slurm-%A_%a.out


set -euo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# python scripts/rsl_rl/train.py \
#    --task=Staircase-G1-Compliance-v0 \
#    --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase \
#    --run_name staircase_compliance_v1 \
#    --video \
#    --video_length 500 \
#    --video_interval 100000


# python scripts/rsl_rl/train.py \
#    --task=Staircase-G1-v0 \
#    --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase \
#    --run_name staircase_baseline \
#    --video \
#    --video_length 500 \
#    --video_interval 100000
