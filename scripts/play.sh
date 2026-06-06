#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --job-name=play_crane
#SBATCH --output=slurm_outputs/slurm-%A_%a.out

set -uo pipefail

cd /move/u/karenvo/Projects/rmr_tracking/

source /move/u/karenvo/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

python scripts/rsl_rl/play_bones.py \
    --task=Bones-Flat-G1-Play-v0 \
    --num_envs=1 \
    --wandb_path robot-mcrobotface/multiclip_bones_popart/9lcilz6r \
    --headless \
    --video \
    --video_length 10000 \
    --popart_head_mode grouped \
    --popart_group_preset all_rewards

