#!/bin/bash
#SBATCH --partition=humanoid  --account=move
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

# baseline
python scripts/rsl_rl/eval.py \
    --task=Staircase-G1-Play-v0 \
    --num_envs=100 \
    --num_episodes=1000 \
    --wandb_path=robot-mcrobotface/staircase/xozov1y9 \
    --headless

python scripts/rsl_rl/play.py \
    --task=Staircase-G1-Play-v0 \
    --num_envs=2 \
    --wandb_path=robot-mcrobotface/staircase/xozov1y9 \
    --headless \
    --video \
    --video_length 30000

# # baseline
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/8ssgkoky \
#     --headless

# # baseline jointpos
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/x09pn0d7 \
#     --headless

# # baseline 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/zmf0n9m2 \
#     --headless

# # baseline jointpos 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/l17spm0a \
#     --headless

# # -------- PUSH --------

# # baseline
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/8ssgkoky \
#     --headless \
#     --push

# # baseline jointpos
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/x09pn0d7 \
#     --headless \
#     --push

# # baseline 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/zmf0n9m2 \
#     --headless \
#     --push

# # baseline jointpos 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/l17spm0a \
#     --headless \
#     --push

# # -------- PUSH FEET --------

# # baseline
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/8ssgkoky \
#     --headless \
#     --push_feet

# # baseline jointpos
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/x09pn0d7 \
#     --headless \
#     --push_feet

# # baseline 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/zmf0n9m2 \
#     --headless \
#     --push_feet

# # baseline jointpos 2step
# python scripts/rsl_rl/eval.py \
#     --task=Staircase-G1-Play-v0 \
#     --num_envs=100 \
#     --num_episodes=1000 \
#     --wandb_path=robot-mcrobotface/staircase_final/l17spm0a \
#     --headless \
#     --push_feet
