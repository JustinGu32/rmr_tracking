#!/bin/bash
#SBATCH --job-name=play_clip
#SBATCH --partition=move  --account=move
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm/play_clip_%j.out
#SBATCH --error=logs/slurm/play_clip_%j.err

mkdir -p logs/slurm

cd /move/u/justingu/rmr_tracking/

source /move/u/justingu/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# ===================== Configuration =====================
ZARR_PATH="/move/data/bones/g1/zarr/locomotion_50hz.zarr"
VIDEO_DIR="eval_results/clip_videos/multiclip_gravity"

# Clip selection (pick ONE)
CLIP_FLAG="--clip_name=jump_and_land_light_003__A001"
# CLIP_FLAG="--clip_name=Jump_002__A017_M"
# CLIP_FLAG="--clip_id=86"

# Policy checkpoints
BONES_WANDB="robot-mcrobotface/multiclip_bones/x6i28m58"
TRACKING_WANDB="robot-mcrobotface/multiclip_bones/3dk71nvh"
# ==========================================================

# # --- Bones compliance policy ---
# python scripts/rsl_rl/play_bones_clip.py \
#     --task=Bones-MultiClip-Compliance-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --wandb_path=${BONES_WANDB} \
#     ${CLIP_FLAG} \
#     --video_dir=${VIDEO_DIR} \
#     --num_envs=1 \
#     --headless \
#     --video

# # --- Tracking baseline policy ---
# python scripts/rsl_rl/play_bones_clip.py \
#     --task=Tracking-MultiClip-Flat-G1-v0 \
#     --zarr_path=${ZARR_PATH} \
#     --wandb_path=${TRACKING_WANDB} \
#     ${CLIP_FLAG} \
#     --video_dir=${VIDEO_DIR} \
#     --num_envs=1 \
#     --headless \
#     --video


GRAVITY_12_WANDB="robot-mcrobotface/multiclip_bones/u90b2ybr"
GRAVITY_6_WANDB="robot-mcrobotface/multiclip_bones/tdhhyrl8"

# --- Bones compliance policy ---
python scripts/rsl_rl/play_bones_clip.py \
    --task=Bones-MultiClip-Compliance-G1-v0 \
    --zarr_path=${ZARR_PATH} \
    --wandb_path=${GRAVITY_12_WANDB} \
    ${CLIP_FLAG} \
    --video_dir=${VIDEO_DIR} \
    --num_envs=1 \
    --headless \
    --video

# --- Bones compliance policy ---
python scripts/rsl_rl/play_bones_clip.py \
    --task=Bones-MultiClip-Compliance-G1-v0 \
    --zarr_path=${ZARR_PATH} \
    --wandb_path=${GRAVITY_6_WANDB} \
    ${CLIP_FLAG} \
    --video_dir=${VIDEO_DIR} \
    --num_envs=1 \
    --headless \
    --video