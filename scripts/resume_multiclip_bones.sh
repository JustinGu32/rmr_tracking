#!/bin/bash
#SBATCH --job-name=multiclip_bones_resume
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/resume_multiclip_bones/resume_%j.out
#SBATCH --error=logs/slurm/resume_multiclip_bones/resume_multiclip_bones_%j.err

set -euo pipefail

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_100hz_walk_jog_jump_all.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_dec4_resumed_dec2 \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 2 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/6kho0snv"

# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_50hz_walk_jog_jump_all.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_dec4_finetune50hz_dec4 \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 4 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/6kho0snv"

# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_100hz_walk_jog_jump_all.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_resumed_dec2 \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 2 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/r9rugj3k"

# -- resume 100hz full --

# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_100hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_100hz_gravity12.81_decimation2_resumed \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 2 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/eworj4ry"

# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_100hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name tracking_100hz_gravity12.81_decimation2_resumed \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 2 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/7x9iqb84"


# -- finetune 33hz full --

# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr  \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_100hz_gravity12.81_decimation2_finetune33hz_dec6 \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 6 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/eworj4ry"

# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name tracking_100hz_gravity12.81_decimation2_finetune33hz_dec6 \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 6 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/7x9iqb84"


# -- finetune33hz walk,jog,jump --

# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_jog_jump_all.zarr  \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_finetune33hz_dec6 \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 6 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/r9rugj3k"


# python scripts/rsl_rl/train_bones.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_jog_jump_all.zarr  \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones \
#     --run_name bones_target_100hz_swish_uniform_walk-jog-jump_noXYterm_G1_29dof_gravity12.81_decimation2_finetune33hz_dec6 \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 6 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones/r9rugj3k"

python scripts/rsl_rl/train_bones.py \
    --task=Tracking-MultiClip-Flat-G1-v0 \
    --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_all.zarr \
    --num_envs=4096 \
    --headless \
    --logger wandb \
    --log_project_name multiclip_bones_standup \
    --run_name tracking_33hz_standup_decimation6_finetuneWalk \
    --ppo_output target \
    --activation swish \
    --double_step \
    --sampling uniform \
    --decimation 6 \
    --wandb_resume "robot-mcrobotface/multiclip_bones_standup/oyqt3l22"

# python scripts/rsl_rl/train_bones.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/u/justingu/rmr_tracking/motions/locomotion_33hz_walk_standup_all.zarr \
#     --num_envs=4096 \
#     --headless \
#     --logger wandb \
#     --log_project_name multiclip_bones_standup \
#     --run_name tracking_33hz_standup_decimation6_finetuneWalkStandup \
#     --ppo_output target \
#     --activation swish \
#     --double_step \
#     --sampling uniform \
#     --decimation 6 \
#     --wandb_resume "robot-mcrobotface/multiclip_bones_standup/oyqt3l22"
