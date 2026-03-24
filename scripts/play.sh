#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=play_staircase
#SBATCH --output=slurm_outputs/slurm-%A_%a.out

set -uo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/rsl_rl/play.py \
    --task=Staircase-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/staircase/xozov1y9 \
    --headless \
    --video \
    --video_length 10000

python scripts/rsl_rl/play.py \
    --task=Staircase-G1-Compliance-Play-v0 \
    --num_envs=1 \
    --wandb_path=robot-mcrobotface/staircase/lk6aq7j0 \
    --headless \
    --video \
    --video_length 10000

