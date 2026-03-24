#!/bin/bash
#SBATCH --partition=move  --account=move
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=eval_staircase
#SBATCH --output=slurm_outputs/slurm-%A_%a.out

set -uo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# -------- NO PUSH --------

python scripts/rsl_rl/eval.py \
    --task=Staircase-G1-Compliance-Play-v0 \
    --num_envs=100 \
    --num_episodes=1000 \
    --wandb_path=robot-mcrobotface/staircase/lk6aq7j0 \
    --headless

# # compliance
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/b3qjmgfs \
#     --headless

# # compliance jointpos
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/0y1k4xs1 \
#     --headless

# # compliance 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/2a3zekuo \
#     --headless

# # compliance jointpos 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/f0mwo11f \
#     --headless

# # -------- PUSH --------

# # compliance
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/b3qjmgfs \
#     --headless \
#     --push

# # compliance jointpos
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/0y1k4xs1 \
#     --headless \
#     --push

# # compliance 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/2a3zekuo \
#     --headless \
#     --push

# # compliance jointpos 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/f0mwo11f \
#     --headless \
#     --push

# # -------- PUSH FEET --------

# # compliance
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/b3qjmgfs \
#     --headless \
#     --push_feet

# # compliance jointpos
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/0y1k4xs1 \
#     --headless \
#     --push_feet

# # compliance 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/2a3zekuo \
#     --headless \
#     --push_feet

# # compliance jointpos 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Compliance-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/f0mwo11f \
#     --headless \
#     --push_feet
