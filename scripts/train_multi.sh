#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --gres=gpu:a5000:2
#SBATCH --job-name=base_tango
#SBATCH --output=slurm_outputs/slurm-%A_%a.out


set -euo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

torchrun --standalone --nproc_per_node=2 scripts/rsl_rl/train.py \
  --task=Tracking-Flat-G1-v0 \
  --registry_name justingu-stanford-university-org/wandb-registry-Motions/tango:latest \
  --headless \
  --logger wandb \
  --log_project_name tango \
  --run_name tango_base_16k \
  --num_envs 16384 \
  --video \
  --video_length 500 \
  --video_interval 10000