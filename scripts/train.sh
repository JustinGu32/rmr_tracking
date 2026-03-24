#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1 
#SBATCH --job-name=compliance0.5
#SBATCH --output=slurm_outputs/slurm-%A_%a.out


set -uo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# python scripts/rsl_rl/train.py \
#    --task=Staircase-G1-Compliance-v0 \
#    --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase_final \
#    --double_step \
#    --run_name staircase_compliance_2step \
#    --resume True \
#    --load_run 2026-03-13_16-02-55_staircase_compliance_2step &

# python scripts/rsl_rl/train.py \
#    --task=Staircase-G1-v0 \
#    --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase_final \
#    --double_step \
#    --run_name staircase_baseline_2step \
#    --resume True \
#    --load_run 2026-03-13_16-02-48_staircase_baseline_2step &

# wait

# python scripts/rsl_rl/train.py \
#    --task=Staircase-G1-Compliance-v0 \
#    --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase_final \
#    --run_name staircase_compliance

# python scripts/rsl_rl/train.py \
#    --task=Staircase-G1-v0 \
#    --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
#    --headless \
#    --logger wandb \
#    --log_project_name staircase_final \
#    --run_name staircase_baseline

python scripts/rsl_rl/train.py \
   --task=Staircase-G1-Compliance-v0 \
   --registry_name robot-mcrobotface/csv_to_npz/staircase_final_v3:latest \
   --headless \
   --logger wandb \
   --log_project_name staircase_final \
   --run_name staircase_compliance0.9