#!/bin/bash
#SBATCH --job-name=eval_multiclip
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/eval_multiclip_%j.out
#SBATCH --error=logs/slurm/eval_multiclip_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/tg334ct9 \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_gravity_swish_uniform \
#     --results_name=bones_target_50hz_swish_gravcur6.7_uniform_pt3 \
#     --activation swish

# Walk+jog eval — mirror of training run #3 (33hz, swish, uniform, no gravity curriculum)
python scripts/rsl_rl/eval_multiclip.py \
    --task=Bones-MultiClip-Compliance-G1-v0 \
    --zarr_path=/move/data/bones/g1/zarr/locomotion_33hz.zarr \
    --wandb_path=robot-mcrobotface/multiclip_bones/c15qko8c \
    --num_envs=16384 \
    --headless \
    --results_dir=eval_results/multiclip_walk_jog_33hz \
    --results_name=bones_target_33hz_swish_uniform_walk-jog \
    --activation swish \
    --include_motion_types walk,jog

# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/3oftz6dh \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_tracking_timetolive \
#     --results_name=tracking_target_50hz_swish_uniform_timetolive \
#     --activation swish

# python scripts/rsl_rl/eval_multiclip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=/move/data/bones/g1/zarr/locomotion_50hz.zarr \
#     --wandb_path=robot-mcrobotface/multiclip_bones/etrf9b1c \
#     --num_envs=16384 \
#     --headless \
#     --results_dir=eval_results/multiclip_tracking_gravity \
#     --results_name=tracking_target_50hz_swish_uniform_gravity_12_pt2 \
#     --activation swish